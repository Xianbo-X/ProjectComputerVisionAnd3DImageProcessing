import numpy as np

class LinearLayer():
    """
    y=xA^T+b
    """
    def __init__(self,input_size,output_size,initialization,bias=False) -> None:
        self.weights=initialization(output_size,input_size)
        self.bias=None
        self.lastData=None
        self.optimizers={}
        self.grad=None
        self.weight_grad=None
        self.bias_grad=None
        self.require_update=True
        if bias:
            self.bias=initialization(output_size,1)
    
    def __call__(self, input):
        return self.forward(input)

    def set_weight(self,weights):
        if weights.shape != self.weights.shape:
            raise BaseException("Shape Error")
        self.weights=weights
    
    def set_bias(self, bias):
        if bias is False:
            raise BaseException("Not allowed to Set bias")
        if bias.shape != self.bias.shape:
            raise BaseException("Shape Error")
        self.bias = bias
    
    def set_optimizer(self,optimizer,forwhom):
        self.optimizers[forwhom]=optimizer
        

    def forward(self,x):
        self.lastData=x
        result=x@self.weights.T
        if not (self.bias is None):
            result=result+self.bias
        return result

    def backward(self,chain_grad,data=None):
        if data is None:
            data=self.lastData
        self.derivative_on_weight(chain_grad,data=data)
        if not (self.bias is None):
            self.derivative_on_bias(chain_grad,data=data)
        self.gradient(chain_grad,data=data)
        return self.grad
    
    def gradient(self,chain_grad,data=None):
        if data is None:
            data=self.lastData
        self.grad=self.derivative_on_input(chain_grad,data)
        return self.grad

    def derivative_on_input(self,chain_grad,data=None):
        return self.derivative_on_input_batches(chain_grad,data)
    
    def derivative_on_input_batches(self,chain_grad,data):
        return chain_grad@self.weights

    def derivative_on_weight(self,chain_grad,reduce="sum",data=None):
        if data is None:
            data=self.lastData
        if (reduce=="sum"):
            result=chain_grad.T@data
        if (reduce=="mean"):
            result=chain_grad.T@data/data.shape[0]
        if (reduce=="none"):
            result=self.derivative_on_weight_batches(chain_grad,data)
        self.weight_grad=result
        return self.weight_grad
    
    def derivative_on_weight_batches(self,chain_grad,data):
        gradient_sample=[]
        for batch_num in range(data.shape[0]):
            gradient_sample.append(self.derivative_on_weight_single(chain_grad[batch_num],data[batch_num]))
        return np.array(gradient_sample)
    
    def derivative_on_weight_single(self,chain_grad,data):
        return np.tensordot(chain_grad,data,0)
    
    def derivative_on_bias(self,chain_grad,reduce="sum",data=None):
        if reduce=="sum":
            result = chain_grad.sum(0)
        if reduce=="mean":
            result = chain_grad.sum(0)/chain_grad.shape[0]
        if reduce=="none":
            result =chain_grad
        self.bias_grad=result
        return self.bias_grad
    
    def derivative_on_bias_single(self,chain_grad,data):
        return chain_grad
        
    def update(self,chain_grad,*arg):
        self.update_weights(chain_grad,*arg)
        if not (self.bias is None):
            self.update_bias(chain_grad,*arg)
    
    def update_weights(self,chain_grad,*arg):
        weights_grad=self.derivative_on_weight(chain_grad,"mean")
        self.set_weight(self.optimizers["weights"](self.weights,weights_grad,*arg))

    def update_bias(self,chain_grad,*arg):
        bias_grad=self.derivative_on_bias(chain_grad,"mean")
        self.set_bias(self.optimizers["bias"](self.bias,bias_grad,*arg))

