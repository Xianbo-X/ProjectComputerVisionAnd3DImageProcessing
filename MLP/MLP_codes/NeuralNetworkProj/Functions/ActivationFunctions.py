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
    # return 1/(1+np.exp(-x))

def sigmoid_derivative(x,chain_grad):
    if len(x.shape)<2: 
        raise "Shape error, expected a 2d array"
    return chain_grad*sigmoid(x)*(1-sigmoid(x))

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

# class ReLu(BasicLayer):
#     def __init__(self, inputNode, outputNode) -> None:
#         super(ReLu,self).__init__(inputNode, outputNode)
    
#     def InitInfo(self):
#         self.name="Layer "+str(BasicLayer.num)
#         self.typeName="ReLu"
    
#     def Output(self,data,recordData=True):
#         if recordData==True:
#             self.lastInput=data
#         return np.maximum(data,0)
        
#     def __call__(self, *args):
#         return self.Output(args[0])

#     def Grad(self,chainGrad,data=None):
#         if data is None:
#             data=self.lastInput
#         return self.GradBatches(chainGrad,data)

#     def GradBatches(self,chainGrad,data):
#         curGrad=1/2*(data/abs(data)+1)
#         return chainGrad*curGrad # Hadamard Product (element-wise product)

#     def DerivativeOfWeight(self,chainGrad,data,summation=False):
#         raise NotImplemented