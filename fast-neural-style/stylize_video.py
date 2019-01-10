import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import skvideo.io
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from fast_neural_style.transformer_net import TransformerNet
from fast_neural_style.utils import recover_image, tensor_normalizer
from tqdm import tqdm

# Preprocess Pipeline
preprocess = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    tensor_normalizer()
])

# Setup the Model Architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerNet()
transformer.to(device)

# Load the Model and Stylize the Content Video
data_path = '/tmp2/vincentwu929/DIP_final/Sky/content/sky2/Video2_HR.avi'
mask_path = '/tmp2/vincentwu929/DIP_final/Sky/mask/mask_HR2.png'
video_name = 'Video2_HR'
style_name = 'ZaoWouKi' #'crayon', 'fountainpen', 'ZaoWouKi'
BATCH_SIZE = 8

mask = cv2.imread(mask_path).astype(np.bool)
save_model_path = "./models/" + style_name + "_10000_unstable_vgg19.pth"
transformer.load_state_dict(torch.load(save_model_path))
transformer.eval()
batch = []
videogen = skvideo.io.FFmpegReader(data_path)
writer = skvideo.io.FFmpegWriter("/tmp2/vincentwu929/DIP_final/" +\
                                 video_name + "_" + style_name + ".avi")
try:
    with torch.no_grad():
        for frame in tqdm(videogen.nextFrame()):
            batch.append(preprocess(Image.fromarray(frame[0:4000,1000:5000])).unsqueeze(0))
            if len(batch) == BATCH_SIZE:
                for frame_out in recover_image(transformer(
                    torch.cat(batch, 0).cuda()).cpu().numpy()):
                    frame_out = cv2.resize(frame_out, (4000, 4000), cv2.INTER_CUBIC)
                    out_img = frame.copy()
                    out_img[0:4000,1000:5000] = frame_out * mask + frame[0:4000,1000:5000] * (~mask)
                    writer.writeFrame(out_img)
                batch = []
except RuntimeError as e:
    print(e)
    pass
writer.close()
