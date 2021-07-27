# model

# layers with 8,16,32,64,128 filters with all 3x3 filter size, stride 1
# 2x2 pooling

# forward pass will be each conv layer with relu activation, finally through a dense with relu


import torch
from torch import tensor
import torch.nn as nn
relu = nn.functional.relu

#we dont need to build the class from scratch

#input shape may be wrong, will have to check spectrogram data
def model_func(input_shape = (288, 432,4), classes = 10):
   input_x =  torch.tensor(input_shape)

   #first convolution
   x = relu(nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x)

   x = relu(nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x)

   x = relu(nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x)

   x = relu(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x)