# TRAINING THE CNN MODEL
# Note: Will most likely have to plot loss and accuracy after
# code since live noggin plots can't be used with this keras setup

from tensorflow.keras.optimizers import Adam
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
    # return model and history
    return (model, history)

"""
ALTERNATIVE CODE (PYTORCH)
--------------------------

from .model import *
import torch.nn.functional.cross_entropy
import torch.optim.Adam
import torch as t

# DATA GENERATION TO BE COMPLETED
train_dir = "recognition/data/spectrograms/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=128)

validation_dir = "recognition/data/testing_data/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)

# THIS PROBABLY WORKS HOPEFULLY MAYBE
def get_f1(y_true, y_pred):
    Returns the f1 distance between two input tensors
@@ -17,6 +31,99 @@ def get_f1(y_true, y_pred):
    float: 
        F1 distance between y_true and y_pred
    true_positive = torch.sum()
    # pathway for computing f1-score given truth and prediction, essentially "accuracy"
    true_positives = t.sum(t.round(t.clip(y_true * y_pred, 0, 1)))
    possible_positives = t.sum(t.round(t.clip(y_true, 0, 1)))
    predicted_positives = t.sum(t.round(t.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives+1e-07)
    recall = true_positives / (possible_positives+1e-07)
    f1_val = 2 * (precision*recall) / (precision+recall+1e-07)
    return f1_val

#normal accuracy
def accuracy(predictions, truthdata):
    Returns the mean classification accuracy for a batch of predictions.
    ''''
    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points
    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)
    
    Returns
    -------
    float
    '''''
    truth = []
    for i in truthdata:
        truth.append(np.argmax(i)) 
    return np.mean(np.argmax(predictions, axis=1) == truth) # <COGLINE>


# TRAINING SETUP
model = Model()
optim = Adam(learning_rate=0.00005)
batch_size = 100
epochs = 20                      #recommended 70


for epoch_cnt in range(epochs):
    idxs = np.arange(len(x_train))  # -> array([0, 1, ..., 9999])
    np.random.shuffle(idxs)  

    for batch_cnt in range(len(x_train)//batch_size):
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_train[batch_indices]  # random batch of our training data

        spectrograms, labels = batch   #may be reversed based on how x_train is formatted

        optim.zero_grad()  #pytorch accumulates gradients

        # compute the predictions for this batch by calling on model
        prediction = model(spectrograms)

        # compute the true (a.k.a desired) values for this batch: 
        #truth = y_train[batch_indices]

        # compute the loss associated with our predictions(use softmax_cross_entropy)
        loss = cross_entropy(prediction, labels) 

        # back-propagate through your computational graph through your loss
        loss.backward()

        # execute gradient descent by calling step() of optim
        optim.step()

        # compute the accuracy between the prediction and the truth 
        acc = accuracy(labels, prediction)

        # set the training loss and accuracy
        plotter.set_train_batch({"loss" : loss.item(),
                                 "accuracy" : acc},
                                 batch_size=batch_size)

    # Here, we evaluate our model on batches of *testing* data
    # this will show us how good our model does on data that 
    # it has never encountered
    # Iterate over batches of *testing* data
    for batch_cnt in range(0, len(x_test)//batch_size):
        idxs = np.arange(len(x_test))
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_test[batch_indices] 

        with mg.no_autodiff:
            # get your model's prediction on the test-batch
            prediction = model(batch)

            # get the truth values for that test-batch
            truth = y_test[batch_indices]

            # compute the test accuracy
            acc = accuracy(prediction, truth)

        # log the test-accuracy in noggin
        plotter.set_test_batch({"accuracy": acc}, batch_size=batch_size)

    plotter.set_train_epoch()
    plotter.set_test_epoch()
plotter.plot()

...
one_hots = {"blues":[1,0,0,0,0,0,0,0,0], "classical":[0,1,0,0,0,0,0,0,0], "country":[0,0,1,0,0,0,0,0,0], "disco":[0,0,0,1,0,0,0,0,0], "hiphop":[0,0,0,0,1,0,0,0,0], "metal":[0,0,0,0,0,1,0,0,0], "pop":[0,0,0,0,0,0,1,0,0], "reggae":[0,0,0,0,0,0,0,1,0],"rock":[0,0,0,0,0,0,0,0,1]}

#how to get the array of img data (hopefully) - all images are same dimensions so should work

images_dir = Path(r"C:/Users/g_bab/Downloads/genre-recognition/spectrograms").expanduser()

X_image_array=[]
for fname in listdir(images_dir):
    fpath = os.path.join(images_dir, fname)
    im = Image.open(fpath)
    X_image_array.append(im)

x_data = []
for x in range(len(X_image_array)):
    X_image=np.array(X_image_array[x],dtype='uint8')
    x_data.append(X_image)
np.stack(x_data)                                                    #will output a shape (A, B, C, D)  - A: number of images, B:len, C: width, D:color channels 

split = int(len(x_data)*0.5)

x_train = x_data[:split]       #train/test split
x_test = x_data[split:]

#read in complete labels list here
labelslist = []
for i in range(600):
    labelslist.append(one_hots["blues"])
for j in range(600):
    labelslist.append(one_hots["classical"])
for j in range(600):
    labelslist.append(one_hots["country"])
for j in range(600):
    labelslist.append(one_hots["disco"])
for j in range(600):
    labelslist.append(one_hots["hiphop"])
for j in range(600):
    labelslist.append(one_hots["metal"])
for j in range(600):
    labelslist.append(one_hots["pop"])
for j in range(600):
    labelslist.append(one_hots["reggae"])
for j in range(600):
    labelslist.append(one_hots["rock"])
print(labelslist[:20])

y_train = labelslist[:split]
y_test = labelslist[split:]

"""