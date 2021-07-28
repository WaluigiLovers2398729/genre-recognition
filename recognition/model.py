# CONVOLUTIONAL NETWORK MODEL
# Note: We're using the keras library because the streamlined functions are more optimized for
# managing memory when dealing with the large amount of files we split from the original GTZAN dataset

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, Dense, Activation, BatchNormalization, Flatten, Dropout,Conv2D, MaxPooling2D)

"""
# directory of training data (spectrograms)
train_dir = "recognition/data/spectrograms/"
# augments image data for batch generation
#   rescale: scales rgb images data down so model can process
train_datagen = ImageDataGenerator(rescale=1./255)
# navigates to directory and generates batches of augmented data
#   target_size: dimensions to resize images (288, 432)
#   color_mode: color channels of images (rgba is 4 channels)
#   class_mode: type of label arrays returned (categorical is 2D one-hot encoded labels)
#   batch_size: size of batches of data (128)
# returns iterable yielding tuples of (x, y), where x is np array containing 
# batch of images and y is np array containing respective batch of labels
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(288,432), color_mode="rgba", class_mode='categorical', batch_size=128)

# repeat same process for validation data (testing_data)
validation_dir = "recognition/data/testing_data/"
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(validation_dir, target_size=(288,432), color_mode='rgba', class_mode='categorical', batch_size=128)
"""

# use the following if testing model & training with smaller samples:
# Note: manually copy a fraction of training/validation data to respective folders
# NORMAL: spectrograms (5400) / testing_data (5400) / genres (600)
# SMALLER: spectrograms (540) / testing_data (540) / genres (60)

train_dir = "recognition/data/temp/temp_train/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(288,432), color_mode="rgba", class_mode='categorical', batch_size=128)
validation_dir = "recognition/data/temp/temp_valid/"
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(validation_dir, target_size=(288,432), color_mode='rgba', class_mode='categorical', batch_size=128)
def GenreModel(input_shape = (288, 432, 4), classes=9):

    """
    docstring
    """
    # instantiate keras tensor
    X_input = Input(input_shape)

    # first convolutional layer
    X = Conv2D(8, kernel_size=(3,3), strides=(1,1))(X_input) # layer
    X = BatchNormalization(axis=3)(X) # normalize
    X = Activation('relu')(X) # relu activation
    X = MaxPooling2D((2,2))(X) # pooling
    
    # second convolutional layer
    X = Conv2D(16, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    
    # third convolutional layer
    X = Conv2D(32, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    # fourth convolutional layer
    X = Conv2D(64, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    
    # fifth convolutional layer
    X = Conv2D(128, kernel_size=(3,3), strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    # flattens x after passing through convolutional layers
    X = Flatten()(X)
    
    # dropout layer randomly sets input units to 0 with freq 
    # of rate at each step during training time, which helps prevent overfitting
    #   rate: fraction of the input units to drop
    X = Dropout(rate=0.3)

    # dense layer with softmax activation to output class probabilities
    X = Dense(classes, activation='softmax', name='fc'+str(classes))(X)

    # groups layers into an object with both training and inference features
    model = Model(inputs=X_input, outputs=X, name='GenreModel')
    return model

"""
ALTERNATIVE CODE (PYTORCH)
--------------------------

import torch
from torch import tensor
import numpy as np
import torch.nn as nn
import torch
relu = nn.functional.relu
softmax = nn.functional.softmax

class Model(nn.Module):

    # initializer function
    def __init__(self, input_shape=(288,432, 4), classes=9):
        docstring
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
        docstring
        x = relu(self.conv1(x))     #after each convolution, pool and normalize
        x = self.pool(x)            #used to reduce dimensionality
        x = self.batchnorm(x)

        x = relu(self.conv2(x))
        x = self.pool(x)
        x = self.batchnorm(x)

        x = relu(self.conv3(x))
        x = self.pool(x)
        x = self.batchnorm(x)

#we dont need to build the class from scratch
        x = relu(self.conv4(x))
        x = self.pool(x)
        x = self.batchnorm(x)

#input shape may be wrong, will have to check spectrogram data
def model_func(input_shape = (288, 432,4), classes = 10):
   input_x =  torch.tensor(input_shape)
        x = relu(self.conv5(x))
        x = self.pool(x)
        x = self.batchnorm(x)

   #first convolution
   x = relu(nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x)
        x = x.flatten()            #dropout is used to help prevent overfitting - it randomly sets inputs to 0 and scales everything up so that the total sum doesn't change
        x = self.dropout(x)

   x = relu(nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x)
        x = softmax(self.final_dense(x))  #converts to final 9 frequency scores

   x = relu(nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x)
        return x 

   x = relu(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1))
   pool = nn.MaxPool2d(2)
   x  = pool(x) 
"""
