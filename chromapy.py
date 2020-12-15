import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from cnn_model import Model

def preprocess_image(img, height=256, width=256):
    """Return the light intensity part of an image, resized and converted to tensor"""
    image = Image.open(img).convert('RGB')
    image_r = image.resize((width, height))
    image_r_np = np.array(image_r) / 255.0
    
    # Convert image to Lab format
    image_lab = color.rgb2lab(image_r_np)
    # Extract L dimension
    image_l = image_lab[:,:,0]
    # Convert to tensor and add relevant dimensions
    tens_l = torch.Tensor(image_l)[None,None,:,:]

    return tens_l

def postprocess_tens(orig_img, ab, mode='bilinear'):
	# orig_img 	1 x 1 x H_orig x W_orig
	# ab 		1 x 2 x H x W

    HW_orig = orig_img.shape[2:]
    HW = ab.shape[2:]

	# Resize if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        ab_orig = F.interpolate(ab, size=HW_orig, mode=mode)
    else:
        ab_orig = ab

    out_lab_orig = torch.cat((orig_img, ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True,
	help="path to input black and white image")
parser.add_argument('--use_gpu', action='store_true', default=False, 
    help='whether to use GPU')
args = parser.parse_args()

preprocessed_tensor = preprocess_image(args.image)

# print(prepocessed_tensor.shape)

model = Model().eval()
ab_out = model.forward(preprocessed_tensor)
print(ab_out.shape)

image_new = postprocess_tens(preprocessed_tensor, ab_out)

plt.imshow(image_new)
plt.show()
