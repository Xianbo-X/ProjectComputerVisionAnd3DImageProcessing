import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import transforms

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

def get_mnist_data(root="./data"):
    """"
    Load Mnist Data by torchvision datasets, apply transfom ToTensor()
    """
    train_data = datasets.MNIST(root=root, train=True, download=True,transform=transforms.ToTensor())
    test_data = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    return train_data,test_data

def get_data_loader(train_data,test_data,batch_size):
    """
    Generate dataloader
    """
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)
    return train_loader,test_loader