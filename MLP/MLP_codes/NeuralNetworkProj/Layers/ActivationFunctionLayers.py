from abc import ABCMeta, abstractmethod
from itertools import chain
from turtle import forward
import numpy as np

class ElementwiseActivation():
    def __init__(self,forward,derivative) -> None:
        self.forward=forward
        self.derivative=derivative
        self.grad=None
        self.require_update=False

    def __call__(self, data):
        self.last_data=data
        return self.forward(data)

    def backward(self,chain_grad,data=None):
        return self.gradient(chain_grad,data)
    
    def gradient(self,chain_grad,data=None):
        if data is None:
            data=self.last_data
        self.grad=self.derivative(data,chain_grad)
        return self.grad


    # return 1/(1+np.exp(-x))



def softmax(x):
    """Calculate the softmax value of x, stablized by minus the maximum value
    
    (exp(x-max(x))/sum{exp(x-max(x))})

    Args:
        x: input variables numpy array in shape (n_samples,n_features)

    Returns:
        An array with the same shape as input. softmax(x)
    """
    if len(x.shape)<2: 
        raise "Shape error, expected a 2d array"
    exp_val=np.exp(x-np.max(x,axis=1,keepdims=True))
    softmax_val=exp_val/np.sum(exp_val,axis=1,keepdims=True)
    return softmax_val
    

def logsoftmax(x):
    """Calculate the log of softmax of x, stablized by minus the maximum value
    
    log( exp(x-max(x))/sum{exp(x-max(x))} )

    Args:
        x: input variables numpy array in shape (n_samples,n_features)

    Returns:
        An array with the same shape as input. logsoftmax(x)
    """
    if len(x.shape)<2: 
        raise "Shape error, expected a 2d array"
    x=x-np.max(x,axis=1,keepdims=True) # stablize softmax
    return x-np.log(np.sum(np.exp(x),axis=1,keepdims=True))

def relu(x):
    return np.maximum(x,0)

def relu_derivative(x,chain_grad):
    return chain_grad*(1/2 * ((x/np.abs(x))+1) )


class ActivationFunctions(metaclass=ABCMeta):
    def __init__(self) -> None:
        # self.reduce=None
        self.require_update=False
        self.grad=None
        self.lastData=None
        pass

    def __call__(self, x):
        self.lastData=x
        return self.forward(self.lastData)

    @abstractmethod
    def forward(self,x):
        pass

    @abstractmethod
    def gradient(self, chain_grad,x):
        pass

    # @abstractmethod
    # def gradient_batches(self,chain_grad,x):
    #     pass

    # @abstractmethod
    # def gradient_single(self,chain_grad,x):
    #     pass

    @abstractmethod
    def backward(self,chain_grad):
        pass

class Softmax(ActivationFunctions):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def softmax(data):
        '''
        data[batches][...]
        '''
        probability = np.exp(data)
        # partitionFn = np.sum(probability, axis=0)
        partitionFn = probability.sum(axis=1).reshape(-1,1)
        # assert (probability.shape[1] == partitionFn.shape[0])
        return probability / partitionFn

    def forward(self,x):
        return self.softmax(x)


    def backward(self,chain_grad):
        self.grad=self.gradient(chain_grad,self.lastData)
        return self.grad

    def gradient(self, chain_grad, data):
        return self.gradient_batches(chain_grad, data)

    def gradient_batches(self,chain_grad,data):
        res=[]
        for batchNum in range(data.shape[0]):
            res.append(self.gradient_single(chain_grad[batchNum], data[batchNum]))
        return np.array(res)
    
    def gradient_single(self, chain_grad, data):
        res=self.softmax(data.reshape(1, -1)).squeeze()
        return np.matmul(chain_grad.reshape(1, -1), res * (np.identity(res.shape[0]) - res.reshape(-1, 1))).squeeze(0)
    
    def derivative_on_weight(self, chain_grad, data, summation=False):
        raise NotImplemented

class ReLu(ActivationFunctions):
    def __init__(self) -> None:
        super(ReLu,self).__init__()
    
    def forward(self,x):
        return self.relu(x)

    def backward(self, chain_grad):
        self.grad=self.gradient(chain_grad,self.lastData)
        return self.grad

    def relu(self,x):
        return np.maximum(x,0)
    

    def gradient(self,chain_grad,x):
        return self.grad_batches(chain_grad,x)

    def grad_batches(self,chain_grad,x):
        cur_grad=1/2*(x/abs(x)+1)
        return chain_grad*cur_grad # Hadamard Product (element-wise product)

class Sigmoid(ActivationFunctions):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        return self.sigmoid(x)
    
    def backward(self, chain_grad):
        self.grad=self.gradient(chain_grad,self.lastData)
        return self.grad
    
    @staticmethod
    def sigmoid(x:np.ndarray):
        """Calculate the sigmoid function of x
        Args:
            x: input variables numpy array in shape (n_samples,n_features)
        Returns:
            An array with the same shape as input. Sigmoid(x)
        """
        result=np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                cur_x=x[i][j]
                if cur_x<0:
                    result[i][j]=np.exp(cur_x)/(1+np.exp(cur_x))
                else:
                    result[i][j]=1/(1+np.exp(-cur_x))
        return result

    def gradient(self, chain_grad, x):
        if len(x.shape)<2: 
            raise "Shape error, expected a 2d array"
        return chain_grad*self.sigmoid(x)*(1-self.sigmoid(x))