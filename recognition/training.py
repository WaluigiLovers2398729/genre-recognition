# TRAINING THE CNN MODEL
# Note: Will most likely have to plot loss and accuracy after
# code since live noggin plots can't be used with this keras setup

from matplotlib import pyplot as plt
from keras.optimizers import Adam
from .model import *
import keras.backend as K

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

def train():
    """
    docstring
    """
    # create model
    model = GenreModel(input_shape=(288, 432, 4), classes=9)
    # create optimizer
    optim = Adam(learning_rate=0.0005)
    # configure the model for training
    model.compile(optimizer = optim, loss='categorical_crossentropy', metrics=['accuracy', get_f1]) 
    # training: fits the model on data yielded batch-by-batch by a python generator
    history = model.fit_generator(train_generator, epochs=70, validation_data=valid_generator)

    # accuracy plotting [probably has to be fixed]
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # loss plotting [probably has to be fixed]
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()