import torch 
import torch.optim.Adam

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
    true_positive = torch.sum()

optim = Adam(learning_rate = 0.0005)
