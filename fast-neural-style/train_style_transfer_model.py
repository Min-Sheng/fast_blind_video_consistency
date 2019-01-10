import os
import argparse
import glob
import cv2
import time 
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch.autograd import Variable

from fast_neural_style.transformer_net import TransformerNet
from fast_neural_style.utils import (
    gram_matrix, recover_image, tensor_normalizer
)
from fast_neural_style.loss_network import LossNetwork

CONTENT_WEIGHT = 1
STYLE_WEIGHTS = np.array([1e-1, 1, 1e1, 5, 1e1]) * 5e4
REGULARIZATION = 1e-6
LOG_INTERVAL = 50
LR = 1e-3
    
 # Utility function to save debug images during training:
def save_debug_image(tensor_orig, tensor_transformed, filename):
    assert tensor_orig.size() == tensor_transformed.size()
    result = Image.fromarray(recover_image(tensor_transformed.cpu().numpy())[0])
    orig = Image.fromarray(recover_image(tensor_orig.cpu().numpy())[0])
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    new_im.paste(orig, (0,0))
    new_im.paste(result, (result.size[0] + 5,0))
    new_im.save(filename)

    
def train(transformer, loss_network, gram_style, gram_matrix, train_loader,\
              content_weight, regularization, style_weights, log_interval,\
              optimizer, device, steps, base_steps=0):
    transformer.train()
    count = 0
    agg_content_loss = 0.
    agg_style_loss = 0.
    agg_reg_loss = 0.   
    while True:
        for x, _ in train_loader:
            count += 1
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)            

            with torch.no_grad():
                xc = x.detach()

            features_y = loss_network(y)
            features_xc = loss_network(xc)

            with torch.no_grad():
                f_xc_c = features_xc[2].detach()

            content_loss = content_weight * mse_loss(features_y[2], f_xc_c)

            reg_loss = regularization * (
                torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            style_loss = 0.
            for l, weight in enumerate(style_weights):
                gram_s = gram_style[l]
                gram_y = gram_matrix(features_y[l])
                style_loss += float(weight) * mse_loss(gram_y, gram_s.expand_as(gram_y))

            total_loss = content_loss + style_loss + reg_loss 
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss
            agg_style_loss += style_loss
            agg_reg_loss += reg_loss

            if count % log_interval == 0:
                mesg = "{} [{}/{}] content: {:.2f}  style: {:.2f}  reg: {:.2f} total: {:.6f}".format(
                            time.ctime(), count, steps,
                            agg_content_loss / log_interval,
                            agg_style_loss / log_interval,
                            agg_reg_loss / log_interval,
                            (agg_content_loss + agg_style_loss + 
                             agg_reg_loss ) / log_interval
                        )
                print(mesg)
                agg_content_loss = 0.
                agg_style_loss = 0.
                agg_reg_loss = 0.
                agg_stable_loss = 0.
                transformer.eval()
                y = transformer(x)
                save_debug_image(x, y.detach(), "./fast-neural-style/debug_{}/{}.png".format(opts.style_name, base_steps + count))
                transformer.train()

            if count >= steps:
                return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Fast Neural Style Transformer")

    # options
    parser.add_argument('-mode',            type=str,     default="train",    choices=["train", "test", "all"],   help='Process mode')
    parser.add_argument('-dataset',         type=str,     default="/home/vincentwu-cmlab/Downloads/coco/",    help='Path of dataset')
    parser.add_argument('-image_size',      type=int,     default=224,              help='Input image size') 
    parser.add_argument('-batch_size',      type=int,     default=4,                help='Batch size')
    parser.add_argument('-seed',            type=int,     default=1080,             help='Number of random seed') 
    parser.add_argument('-style_name',      type=str,     default="crayon",         help='Name of sytle')
    parser.add_argument('-style_image',     type=str,     default="./fast-neural-style/style_images/1.jpg",        help='Path of style image')
    parser.add_argument('-data_path',       type=str,     default="/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky1/Video1_HR.avi",           help='Path of the test data')
    parser.add_argument('-with_mask',            action='store_true',     help='Apply mask?')
    parser.add_argument('-mask_path',       type=str,     default="/home/vincentwu-cmlab/Downloads/DIP_final/Sky/mask/mask_HR1.png",           help='Path of the mask data')
    parser.add_argument('-output_name',     type=str,     default="Video1_HR",  help='Name of the ouput file')
   
    opts = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TransformerNet()
    transformer.to(device)
    
    if opts.mode == 'train' or opts.mode == 'all':
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(opts.seed)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            kwargs = {'num_workers': 4, 'pin_memory': True}
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            kwargs = {}

        # Dataloader
        transform = transforms.Compose([
            transforms.Resize(opts.image_size), 
            transforms.CenterCrop(opts.image_size),
            transforms.ToTensor(), tensor_normalizer()])
        # http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
        train_dataset = datasets.ImageFolder(opts.dataset, transform)
        # http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, **kwargs)

        # Setup Model
        with torch.no_grad():
            loss_network = LossNetwork()
            loss_network.to(device)
        loss_network.eval()

        # Load Style Target
        style_img = Image.open(opts.style_image).convert('RGB')
        with torch.no_grad():
            style_img_tensor = transforms.Compose([
                transforms.Resize(opts.image_size* 2),
                transforms.ToTensor(),
                tensor_normalizer()]
            )(style_img).unsqueeze(0)
            style_img_tensor = style_img_tensor.to(device)

        # Precalculate Gram Matrices of the Style Image
        # http://pytorch.org/docs/master/notes/autograd.html#volatile
        with torch.no_grad():
            style_loss_features = loss_network(style_img_tensor)
            gram_style = [gram_matrix(y) for y in style_loss_features]
        print('# of VGG-19 layers which style loss use:', style_loss_features._fields)

        #for i in range(len(style_loss_features)):
        #    tmp = style_loss_features[i].cpu().numpy()
        #    print(i, np.mean(tmp), np.std(tmp))

        #for i in range(len(style_loss_features)):
        #    print(i, gram_style[i].numel(), gram_style[i].size())

        # Train the Transformer
        torch.set_default_tensor_type('torch.FloatTensor')
        mse_loss = torch.nn.MSELoss()
        # l1_loss = torch.nn.L1Loss()
    
        if not os.path.exists('./fast-neural-style/debug_{}'.format(opts.style_name)):
            os.makedirs('./fast-neural-style/debug_{}'.format(opts.style_name))

        optimizer = Adam(transformer.parameters(), LR * 0.5)

        train(transformer, loss_network, gram_style, gram_matrix, train_loader,\
                  CONTENT_WEIGHT, REGULARIZATION, STYLE_WEIGHTS, LOG_INTERVAL,\
                  optimizer, device, steps=1000, base_steps=0)

        train(transformer, loss_network, gram_style, gram_matrix, train_loader,\
                  CONTENT_WEIGHT, REGULARIZATION, STYLE_WEIGHTS, LOG_INTERVAL,\
                  optimizer, device, steps=3000, base_steps=1000)

        save_model_path = "./fast-neural-style/models/{}_4000_unstable_vgg19.pth".format(opts.style_name)
        torch.save(transformer.state_dict(), save_model_path)


        optimizer = Adam(transformer.parameters(), LR * 0.1)

        train(transformer, loss_network, gram_style, gram_matrix, train_loader,\
                  CONTENT_WEIGHT, REGULARIZATION, STYLE_WEIGHTS, LOG_INTERVAL*2,\
                  optimizer, device, steps=6000, base_steps=4000)

        save_model_path = "./fast-neural-style/models/{}_10000_unstable_vgg19.pth".format(opts.style_name)
        torch.save(transformer.state_dict(), save_model_path)

        # Stylize the content images
    if opts.mode == 'test':
        save_model_path = "./fast-neural-style/models/{}_10000_unstable_vgg19.pth".format(opts.style_name)
        transformer.load_state_dict(torch.load(save_model_path))
        transformer = transformer.eval()
        video = cv2.VideoCapture(opts.data_path)
        if opts.with_mask:
            mask = cv2.imread(opts.mask_path).astype(np.bool)

        frames=[]
        i=0
        while (video.isOpened()):
            ret, frame = video.read()
            if ret == True:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            else:
                break
        video.release()

        transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.ToTensor(),
            tensor_normalizer()])

        frame = frames[0][0:4000,1000:5000]
        img = Image.fromarray(frame)
        img_tensor = transform(img).unsqueeze(0)

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            img_output = transformer(Variable(img_tensor))

        profile_img = recover_image(img_output.data.cpu().numpy())[0]
        profile_img = cv2.resize(profile_img, (4000, 4000), cv2.INTER_CUBIC)

        if opts.with_mask:
            out_img = frames[0]
            out_img[0:4000,1000:5000] = profile_img * mask + frames[0][0:4000,1000:5000] * (~mask)
            output = Image.fromarray(out_img)
        else:
            output = Image.fromarray(profile_img)

        if not os.path.exists('./fast-neural-style/results'):
            os.makedirs('./fast-neural-style/results')
        output.save('./fast-neural-style/results/frame_00000_{}_{}-stylized.jpg'.format(opts.output_name, opts.style_name))
