import imp
import numpy as np
import abc
from abc import ABCMeta, abstractmethod


class LossFuncitonLayer(metaclass=ABCMeta):
    def __init__(self, reduce="mean") -> None:
        self.reduce = reduce
        self.requre_update = False
        self.grad=None
        self.lastData=None
        self.lastLabel=None

    @abstractmethod
    def gradient(self, x, label):
        pass

    def backward(self):
        self.grad=self.gradient(self.lastData,self.lastLabel)
        return self.grad

    def __call__(self, x, label):
        return self.forward(x,label)

    @abstractmethod
    def forward(self,x,label):
        pass

class CrossEntropy(LossFuncitonLayer):
    def __init__(self, reduce="mean") -> None:
        super().__init__(reduce)
    
    def forward(self,x,label):
        self.lastData=x
        self.lastLabel=label
        return self.cross_entropy(x,label,self.reduce)

    def cross_entropy(self, x, label: np.ndarray, reduce="mean"):
        one_hot = np.identity(x.shape[1]).take(label, axis=0)
        return self.cross_entropy_one_hot(x, one_hot, reduce)

    def cross_entropy_one_hot(self, x, onehot_label, reduce="mean"):
        log_data = -np.log(x)
        result = log_data[onehot_label.astype(bool)]
        if reduce == "mean":
            result = np.mean(result)
        return result

    def cross_entropy_one_hot_derivative(self, x, onehot_label, reduce="mean"):
        """"
        """
        # log_data = -np.log(x)
        # result = np.zeros_like(log_data)
        result = np.zeros_like(x)
        result[onehot_label.astype(bool)]=-1/(x[onehot_label.astype(bool)])
        # result=onehot_label/x
        # TODO: solve zero log_data problem
        if reduce == "mean":
            result = result/x.shape[0]
        if reduce == "none":
            raise NotImplemented
        return result

    def cross_entropy_derivative(self, x, label, reduce="mean"):
        one_hot = np.identity(x.shape[1]).take(label, axis=0)
        return self.cross_entropy_one_hot_derivative(x, one_hot, reduce)

    def gradient(self, x, label):
        return self.cross_entropy_derivative(x, label, self.reduce)
