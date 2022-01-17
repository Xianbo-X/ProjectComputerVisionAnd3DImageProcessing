import numpy as np


def normal_initializer(size1, size2):
    return np.random.normal(size=(size1, size2))

# TO DO
def heWeigth(n):
    # calculate the range for the weights
    std = sqrt(2.0 / n)
    # generate random numbers
    numbers = random(1000)
    # scale to the desired range
    scaled = numbers * std
    # summarize
    return scaled.mean()
