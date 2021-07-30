# TRAINING THE CNN MODEL
# Note: Will most likely have to plot loss and accuracy after
# code since live noggin plots can't be used with this keras setup

# (EPOCHS) 25
# (BATCH-SIZE) 100
# (LEARNING RATE) 0.001

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import convert_to_tensor
from .model import *
import keras.backend as K
import numpy as np
import pickle
import os
genres = ('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock')

# <<< VERSION ONE >>> (PNGS)
"""
# directory of training data (spectrograms)
train_dir = "recognition/data/spectrograms/"
# augments image data for batch generation
#   rescale: scales rgb images data down so model can process
train_datagen = ImageDataGenerator(rescale=1./255)
# navigates to directory and generates batches of augmented data
#   target_size: dimensions to resize images (288, 432)
#   color_mode: color channels of images (rgb is 4 channels)
#   class_mode: type of label arrays returned (categorical is 2D one-hot encoded labels)
#   batch_size: size of batches of data (100)
# returns iterable yielding tuples of (x, y), where x is np array containing 
# batch of images and y is np array containing respective batch of labels
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(288,432), color_mode="rgb", class_mode='categorical', batch_size=100)
# repeat same process for validation data (testing_data)
validation_dir = "recognition/data/testing_data/"
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(validation_dir, target_size=(288,432), color_mode='rgb', class_mode='categorical', batch_size=100)
"""

def get_f1(y_true, y_pred): 
    """
    docstring
    """
    # computing accuracy (f1)
    true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives+K.epsilon())
    recall = true_positives / (possible_positives+K.epsilon())
    f1_val = 2 * (precision*recall) / (precision+recall+K.epsilon())
    return f1_val

# <<< VERSION ONE >>> (PNGS)
"""
def train():
    # create model
    model = GenreModel(input_shape=(288, 432, 3), classes=9)
    # create optimizer
    optim = Adam(learning_rate=0.001)
    # configure the model for training
    model.compile(optimizer = optim, loss='categorical_crossentropy', metrics=['accuracy', get_f1]) 
    # training: fits the model on data yielded batch-by-batch by a python generator
    history = model.fit_generator(train_generator, epochs=100, validation_data=valid_generator)
    # saves history as pickle dictionary
    with open('recognition/data/history_backup/history_dict.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # return model and history
    return (model, history)
"""

# <<< VERSION TWO >>> (ARRAYS)
def train():
    one_hots = {"blues":[1,0,0,0,0,0,0,0,0], "classical":[0,1,0,0,0,0,0,0,0], 
            "country":[0,0,1,0,0,0,0,0,0], "disco":[0,0,0,1,0,0,0,0,0], 
            "hiphop":[0,0,0,0,1,0,0,0,0], "metal":[0,0,0,0,0,1,0,0,0], 
            "pop":[0,0,0,0,0,0,1,0,0], "reggae":[0,0,0,0,0,0,0,1,0],
            "rock":[0,0,0,0,0,0,0,0,1]}
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for g in genres:
        print("Training: " + str(g))
        # iterate through the newly sliced 5400 five-second sound files
        for filename in os.listdir(os.path.join("recognition/data/spectrograms/", f"{g}")):
            # training data
            filedata = np.load(f"recognition/data/spectrograms/{g}/{filename}")
            filedata = np.resize(filedata,(128, 216))  
            x_train.append(filedata)
            # training labels
            y_train.append(one_hots[g])
        print("Validation: " + str(g))
        # iterate through the newly sliced 5400 five-second sound files
        for filename in os.listdir(os.path.join("recognition/data/testing_data/", f"{g}")):     
            # testing data
            filedata = np.load(f"recognition/data/testing_data/{g}/{filename}")
            filedata = np.resize(filedata,(128, 216))  
            x_test.append(filedata)
            # testing labels
            y_test.append(one_hots[g])
    #x_train = np.concatenate([x_train[i] for i in range(len(x_train))])  
    print(len(x_train))
    x_train = np.array(x_train)
    print(x_train.shape)
    y_train = np.array(y_train)
    print(y_train)
    print(len(x_train))
    x_test = np.array(x_test)
    print(x_train.shape)
    y_test = np.array(y_test)
    print(y_train)
    validation_data=(x_test, y_test)
    #x_train = np.array(x_train)
    #y_train = np.array(y_train)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    #x_train = convert_to_tensor(x_train)
    #y_train = convert_to_tensor(y_train)
    #x_test = convert_to_tensor(x_test)
    #y_test = convert_to_tensor(y_test)
    
    # create model
    model = GenreModel(input_shape=(128, 216, 1), classes=9)
    # create optimizer
    optim = Adam(learning_rate=0.001)
    # configure the model for training
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy', get_f1]) 
    # training: fits the model on data yielded batch-by-batch by a python generator
    print(type(x_train))
    print(type(y_train))
    print(type(x_test))
    print(type(y_test))
    print(type(validation_data))
    history = model.fit(x_train, y_train, batch_size=100, epochs=25, validation_data=validation_data)
    # saves history as pickle dictionary
    with open('recognition/data/history_backup/history_dict.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # return model and history
    return (model, history)