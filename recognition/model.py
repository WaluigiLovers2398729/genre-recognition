# CONVOLUTIONAL NETWORK MODEL
# Note: We're using the keras library because the streamlined functions are more optimized for
# managing memory when dealing with the large amount of files we split from the original GTZAN dataset

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (Input, Dense, Activation, BatchNormalization, Flatten, Dropout, Conv2D, MaxPooling2D)

def GenreModel(input_shape=(128, 216, 1), classes=9):
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
    # X = Conv2D(64, kernel_size=(3,3), strides=(1,1))(X)
    # X = BatchNormalization(axis=-1)(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((2,2))(X)
    
    # fifth convolutional layer
    # X = Conv2D(128, kernel_size=(3,3), strides=(1,1))(X)
    # X = BatchNormalization(axis=-1)(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((2,2))(X)

    # flattens x after passing through convolutional layers
    X = Flatten()(X)
 
    # dropout layer randomly sets input units to 0 with freq 
    # of rate at each step during training time, which helps prevent overfitting
    #   rate: fraction of the input units to drop
    dropout_layer = Dropout(rate=0.3)
    X = dropout_layer(X, training=True)

    # dense layer with softmax activation to output class probabilities
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)

    # groups layers into an object with both training and inference features
    model = Model(inputs=X_input, outputs=X, name='GenreModel')
    return model