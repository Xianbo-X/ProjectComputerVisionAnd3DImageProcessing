import numpy as np
from NeuralNetworkProj.Functions.ActivationFunctions import sigmoid

class LossFunction():
    def __init__(self,forward,gradient) -> None:
        self.forward=forward
        self.gradient=gradient
        self.require_update=False
    
    def __call__(self, x,label):
        return self.forward(x,label)

def binary_cross_entropy(predicted_op,true_y,reduce="mean"):
    """
    Args:
        predicted_op: 2D array with shape (n_samples,1)
        true_y: a vector or a column array of true labels
        reduce: ("mean"|"sum"|"none") default is mean, the reduction method to reduce across samples
    Returns:
        Cross-entropy value
    """
    ce=-true_y*np.log(predicted_op)-(1-true_y)*np.log(1-predicted_op)
    if (reduce=="mean"):
        ce=np.sum(ce,axis=0,keepdims=True)/ce.shape[0]
    if (reduce=="sum"):
        ce=np.sum(ce,axis=0,keepdims=True)
    return ce

def binary_cross_entropy_derivative(predicted_op,true_y,reduce="mean"):
    """
    Args:
        predicted_op: 2D array with shape (n_samples,1)
        true_y: a vector or a column array of true labels
        reduce: ("mean"|"sum"|"none") default is mean, the reduction method to reduce across samples
    Returns:
        Derivative of binary-CE over x
    """
    assert predicted_op.shape[1]==1, "(n_features,1) required, got "+str(predicted_op.shape)
    gradient=(predicted_op-true_y)/(predicted_op*(1-predicted_op))
    if (reduce=="mean"):
        gradient=gradient/gradient.shape[0]
    return gradient

def binary_cross_entropy_sigmoid(predicted_op,true_y,reduce="mean"):
    """
    """
    true_y=true_y.reshape(-1,1)
    assert (true_y.shape[0]==predicted_op.shape[0])
    return binary_cross_entropy(sigmoid(predicted_op),true_y,reduce)
    

def binary_cross_entropy_sigmoid_derivative(predicted_op,true_y,reduce="mean"):
    """
    Args:
        predicted_op: 2D array with shape (n_samples,1)
        true_y: a vector or a column array of true labels
        reduce: ("mean"|"sum"|"none") default is mean, the reduction method to reduce across samples
    Returns:
        Derivative of BCE(sigmoid(x)) over x
    """
    true_y=true_y.reshape(-1,1)
    assert (true_y.shape[0]==predicted_op.shape[0])
    gradient=sigmoid(predicted_op)-true_y
    if (reduce=="mean"):
        gradient=gradient/gradient.shape[0]
    if (reduce=="sum"):
        gradient=gradient
    return gradient
    