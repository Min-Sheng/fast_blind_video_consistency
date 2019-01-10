import os
import cv2
import argparse
import skvideo.io
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from fast_neural_style.transformer_net import TransformerNet
from fast_neural_style.utils import recover_image, tensor_normalizer
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Infernece Fast Neural Style Transformer")

    # options
    parser.add_argument('-image_size',      type=int,     default=1024,             help='Input image size') 
    parser.add_argument('-batch_size',      type=int,     default=8,                help='Batch size')
    parser.add_argument('-style_name',      type=str,     default="crayon",         help='Name of sytle')
    parser.add_argument('-video_path',       type=str,     default="/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky1/Video1_HR.avi",           help='Path of the test video')
    parser.add_argument('-with_mask',            action='store_true',     help='Apply mask?')
    parser.add_argument('-mask_path',       type=str,     default="/home/vincentwu-cmlab/Downloads/DIP_final/Sky/mask/mask_HR1.png",           help='Path of the mask data')
    parser.add_argument('-output_name',     type=str,     default="Video1_HR",  help='Name of the ouput file')
   
    opts = parser.parse_args()
    
    # Preprocess Pipeline
    preprocess = transforms.Compose([
        transforms.Resize(opts.image_size),
        transforms.ToTensor(),
        tensor_normalizer()
    ])

    # Setup the Model Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TransformerNet()
    transformer.to(device)

    # Load the Model and Stylize the Content Video
    save_model_path = "./fast-neural-style/models/{}_10000_unstable_vgg19.pth".format(opts.style_name)
    transformer.load_state_dict(torch.load(save_model_path))
    transformer.eval()
    
    videogen = skvideo.io.FFmpegReader(opts.video_path)
    if opts.with_mask:
        mask = cv2.imread(opts.mask_path)
        mask = cv2.resize(mask, (opts.image_size, opts.image_size), cv2.INTER_NEAREST).astype(np.bool)
        
    output_stylized_dir = './data/test/processed/fast-neural-style/{}/Sky/{}'.format(opts.style_name, opts.output_name) 
    if not os.path.exists(output_stylized_dir):
        os.makedirs(output_stylized_dir)
    output_original_dir = './data/test/input/Sky/{}'.format(opts.output_name) 
    if not os.path.exists(output_original_dir):
        os.makedirs(output_original_dir)
    
    i=0
    batch = []
    try:
        with torch.no_grad():
            for frame in tqdm(videogen.nextFrame()):
                batch.append(preprocess(Image.fromarray(frame[0:4000,1000:5000])).unsqueeze(0))
                if len(batch) == opts.batch_size:
                    for frame_out in recover_image(transformer(
                        torch.cat(batch, 0).cuda()).cpu().numpy()):
                        
                        out_img = frame_out
                        frame_in = cv2.resize(frame[0:4000,1000:5000], (opts.image_size, opts.image_size), cv2.INTER_CUBIC)

                        if opts.with_mask:
                            out_img = out_img * mask
                            frame_in = frame_in * mask

                        output_stylized = Image.fromarray(out_img)
                        output_stylized.save(os.path.join(output_stylized_dir,'%05d.jpg'%(i)))
                        output_original = Image.fromarray(frame_in)
                        output_original.save(os.path.join(output_original_dir,'%05d.jpg'%(i)))
                        i+=1
                    batch = []
    
    except RuntimeError as e:
        print(e)
        pass
