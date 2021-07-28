# CONVOLUTIONAL NETWORK MODEL
# Note: We're using the keras library because the streamlined functions are more optimized for
# managing memory when dealing with the large amount of files we split from the original GTZAN dataset

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, Dense, Activation, BatchNormalization, Flatten, Dropout,Conv2D, MaxPooling2D)

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

