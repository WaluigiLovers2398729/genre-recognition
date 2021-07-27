# DONE IN JUPYTER NOTEBOOK

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
    """
    Returns the f1 distance between two input tensors

    Parameters
    ---------
    y_true: tensor
        the truth values
    y_pred: tensor
        the values predicted by the model

    Returns
    ------
    float: 
        F1 distance between y_true and y_pred
    """
    # pathway for computing f1-score given truth and prediction, essentially "accuracy"
    true_positives = t.sum(t.round(t.clip(y_true * y_pred, 0, 1)))
    possible_positives = t.sum(t.round(t.clip(y_true, 0, 1)))
    predicted_positives = t.sum(t.round(t.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives+1e-07)
    recall = true_positives / (possible_positives+1e-07)
    f1_val = 2 * (precision*recall) / (precision+recall+1e-07)
    return f1_val

#normal accuracy
def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.
    
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
    """
    return np.mean(np.argmax(predictions, axis=1) == truth) # <COGLINE>

# TRAINING SETUP
model = Model()
optim = Adam(learning_rate=0.00005)
batch_size = 100
epochs = 20


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
        loss = cross_entropy(prediction, labels) #pytorchs implementation isnt hte same as normal categorical cross entropy i dont think, im looking into it

        # back-propagate through your computational graph through your loss
        loss.backward()

        # execute gradient descent by calling step() of optim
        optim.step()
        
        # compute the accuracy between the prediction and the truth 
        acc = get_f1(labels, prediction)
        
        # set the training loss and accuracy
        '''plotter.set_train_batch({"loss" : loss.item(),
                                 "accuracy" : acc},
                                 batch_size=batch_size)'''
    
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
