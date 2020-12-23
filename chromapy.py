import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from cnn_model import Model
from cnn_model2 import Model as Model_unet
import pickle
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True,
        help="path to input black and white image")
    parser.add_argument('--use_gpu', action='store_true', default=False, 
        help='whether to use GPU')
    return parser.parse_args()

def preprocess_training_set(train):
    processed_x = []
    processed_y = []
    for image in train:
        l, ab = preprocess_image(image)
        processed_x.append(l)
        processed_y.append(ab)
    return processed_x, processed_y

def preprocess_image(img, height=256, width=256):
    """Return the light intensity part of an image, resized and converted to tensor"""
    # image = Image.open(img).convert('RGB')
    # image_r = image.resize((width, height))
    image_r_np = np.array(img) / 255.0
    # Convert image to Lab format
    image_lab = color.rgb2lab(image_r_np)
    # Extract L dimension
    image_l = image_lab[:,:,0]
    image_ab = image_lab[:,:,1:]
    # Convert to tensor and add relevant dimensions
    image_l = image_l[None,:,:]

    return image_l, image_ab

def postprocess_tens(orig_img, ab, mode='bilinear'):
	# orig_img 	1 x 1 x H_orig x W_orig
	# ab 		1 x 2 x H x W
    HW_orig = orig_img.shape[2:]
    HW = ab.shape[2:]
    
    print(orig_img.shape)

	# Resize if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        ab_orig = F.interpolate(ab, size=HW_orig, mode=mode)
    else:
        ab_orig = ab

    out_lab_orig = torch.cat((orig_img, ab_orig), dim=1)
    out_lab_orig = out_lab_orig.data.cpu().numpy()
    return color.lab2rgb(out_lab_orig.transpose((0,2,3,1)))

args = parse_arguments()
# image_dict = unpickle('C:\\Users\\karee\\Desktop\\ChromaPy\\data\\cifar-10-python\\cifar-10-batches-py\\data_batch_1')
# print(image_dict[b'data'])
(X, y), (x_test, y_test) = cifar10.load_data()

# Split data into training and validation
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

og_image = x_train[0:10]

x_train, y_train = preprocess_training_set(x_train[:10])
x_val, y_val = preprocess_training_set(x_val[:10])

tensor_x_train = torch.Tensor(x_train).float()
tensor_x_val = torch.Tensor(x_val).float()
tensor_y_train = torch.Tensor(y_train).permute(0,3,1,2).float()
tensor_y_val = torch.Tensor(y_val).permute(0,3,1,2).float()

# Dataset dictionary
dsets = {
    "train": data.TensorDataset(tensor_x_train,tensor_y_train),
    "val": data.TensorDataset(tensor_x_val,tensor_y_val)}

dataloaders = {x : data.DataLoader(dsets[x], batch_size=6, shuffle=True)
                for x in ['train', 'val']}

dataset_sizes = {x : len(dsets[x]) for x in ["train","val"]}

# preprocessed_tensor = preprocess_image(args.image)
model = Model()
# model_unet = Model_unet(1,2)

# model_unet_ft = model_unet.fit(dataloaders,1)
# ab_out = model_unet_ft.forward(tensor_x_train[0:5])

model_ft = model.fit(dataloaders, 1)
ab_out = model_ft.forward(tensor_x_train[0:5])
# print(ab_out.shape)

image_new = postprocess_tens(tensor_x_train[0:5], ab_out)

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(og_image[0])
axarr[0,1].imshow(image_new[0])
axarr[1,0].imshow(og_image[1])
axarr[1,1].imshow(image_new[1])
plt.show()