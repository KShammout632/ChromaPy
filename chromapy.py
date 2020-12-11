import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from cnn_model import Model

def preprocess_image(img, height=256, width=256):
    """Return the light intensity part of an image, resized"""
    image = Image.open(img).convert('RGB')
    image_r = image.resize((width, height))
    image_r_np = np.array(image_r) / 255.0
    
    # Convert image to Lab format
    image_lab = color.rgb2lab(image_r_np)
    # Extract L dimension
    image_l = image_lab[:,:,0]
    # Convert to tensor and add relevant dimensions
    tens_l = torch.Tensor(image_l)[None,None,:,:]
    
    # lab_temp = np.zeros(image_lab.shape)
    # lab_temp[:,:,0] = image_lab[:,:,0]
    # image_lab = color.lab2rgb(lab_temp)
       
    return tens_l

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True,
	help="path to input black and white image")
parser.add_argument('--use_gpu', action='store_true', default=False, 
    help='whether to use GPU')
args = parser.parse_args()

prepocessed_tensor = preprocess_image(args.image)

# plt.imshow(L)
# plt.show()
# plt.imsave("image.png", L)
