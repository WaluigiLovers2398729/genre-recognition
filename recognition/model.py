# CONVOLUTIONAL NEURAL NETWORK MODEL:
# > 5 Convolutional Layers
# > 1 Dropout Layer (to avoid overfitting)
# > 1 Dense Layer w/ Softmax Activation (outputting probabilities)

from torch import tensor
import numpy as np
import torch.nn as nn
import torch
relu = nn.functional.relu
softmax = nn.functional.softmax

class Model(nn.Module):
    
    # initializer function
    def __init__(self, input_shape=(288,432, 4), classes=9):
        """
        docstring
        """
        super(Model, self).__init__()
        # five convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=8, kernel_size=3, stride=1)       
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1)
        # one dropout layer
        self.dropout = nn.Dropout(0.3)
        # one dense layer
        self.final_dense = nn.Linear(128, classes) 
        
        self.batchnorm = nn.BatchNorm2d

        for m in (self.conv1, self.conv2, self.conv3, self.conv4, self.final_dense):   #converts from default weight normalization to glorot(xavier)
            nn.init.constant_(m.bias,0)
            nn.init.xavier_normal_(m.weight, np.sqrt(2))

    # forward-pass function
    def forward(self, x):
        """
        docstring
        """
        x = relu(self.conv1(x))     #after each convolution, pool and normalize
        x = self.pool(x)            #used to reduce dimensionality
        x = self.batchnorm(x)

        x = relu(self.conv2(x))
        x = self.pool(x)
        x = self.batchnorm(x)

        x = relu(self.conv3(x))
        x = self.pool(x)
        x = self.batchnorm(x)

        x = relu(self.conv4(x))
        x = self.pool(x)
        x = self.batchnorm(x)

        x = relu(self.conv5(x))
        x = self.pool(x)
        x = self.batchnorm(x)

        x = x.flatten()            #dropout is used to help prevent overfitting - it randomly sets inputs to 0 and scales everything up so that the total sum doesn't change
        x = self.dropout(x)

        x = softmax(self.final_dense(x))  #converts to final 9 frequency scores

        return x 

