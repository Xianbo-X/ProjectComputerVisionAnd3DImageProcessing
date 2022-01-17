import numpy as np

def make_batches(x_train:np.ndarray,y_train:np.ndarray,batch_size:int,strict_split=True)->list:
    """
    Args:
        x_train: data array with shape (n_samples,n_features).
        y_train: label array with shape (n_samples,1)
        batch_size: integer which is the batch size.
        strict_split: bool, When True,Some data might be droped to keep each batch has exactly batch_size samples. When false, the last batch may contain more samples.
    Returns:
        List of Splited arraies with shape (m_batches,n_samples,n_features)
    """
    batch_nums=x_train.shape[0]//batch_size # integer division
    if (batch_nums*batch_size<x_train.shape[0]):
        print("WARNING: batch_size can't be divided by the number of samples")
    length=x_train.shape[0]
    if strict_split:
        length=batch_size*batch_nums
    x_batches=np.array_split(x_train[:length],batch_nums)
    y_batches=np.array_split(y_train[:length],batch_nums)
    return x_batches,y_batches