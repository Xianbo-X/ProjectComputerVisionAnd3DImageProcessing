import numpy as np

def MinMaxNormalization(data:np.ndarray,shift=0):
    """ Perform min max normalization
    Args:
        data: An array with shape (n_samples,n_features).
    Returns:
        Normalized data array with the same shape as input.
    """
    max_data=np.max(data,axis=0)
    min_data=np.min(data,axis=0)
    return ((data-min_data)/(max_data-min_data)-shift)

def ZeroMeanUnitVarianceNormalization(data:np.ndarray):
    """ Perform Central unit variance normalization
    Args:
        data: An array with shape (n_samples,n_features).
    Returns:
        Normalized data array with the same shape as input.
    """
    mean_data=np.mean(data,axis=0)
    std_data=np.std(data,axis=0)
    return (data-mean_data)/(std_data)