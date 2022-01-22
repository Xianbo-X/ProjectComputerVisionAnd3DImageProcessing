import numpy as np

class LinearLayer():
    """
    y=xA^T+b
    """
    def __init__(self,input_size,output_size,initialization,bias=False) -> None:
        self.weight=initialization(output_size,input_size)
        self.bias=None
        self.trainable_variable_names=["weight"]
        if bias:
            self.trainable_variable_names.append("bias")
            self.bias=initialization(1,output_size)

        self.lastData=None
        self.optimizers={}
        self.grad=None
        self.weight_grad=None
        self.bias_grad=None
        self.require_update=True
    
        self.trainable_variable_seters={}
        for name in self.trainable_variable_names:
            self.trainable_variable_seters[name]=getattr(self,"set_"+name)
    
    def __call__(self, x):
        self.lastData=x
        return self.forward(self.lastData)

    def set_weight(self,weight):
        if weight.shape != self.weight.shape:
            raise BaseException("Shape Error")
        self.weight=weight
        self.weight_grad=None
    
    def set_bias(self, bias):
        if bias is False:
            raise BaseException("Not allowed to Set bias")
        if bias.shape != self.bias.shape:
            raise BaseException("Shape Error")
        self.bias = bias
        self.bias_grad=None
    
    def set_optimizer(self,optimizer,forwhom):
        self.optimizers[forwhom]=optimizer
        

    def forward(self,x):
        return self.linear_transform(x)
    
    def linear_transform(self,x):
        result=x@self.weight.T
        if not (self.bias is None):
            result=result+self.bias
        return result

    def backward(self,chain_grad):
        if not ((self.weight_grad is None) or (self.bias_grad is None)):
            raise ("Can't backward twice before update parameters")

        data=self.lastData
        # if data is None:
            # data=self.lastData
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
        return chain_grad@self.weight

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
        weight_grad=self.derivative_on_weight(chain_grad,"mean")
        self.set_weight(self.optimizers["weights"](self.weight,weight_grad,*arg))

    def update_bias(self,chain_grad,*arg):
        bias_grad=self.derivative_on_bias(chain_grad,"mean")
        self.set_bias(self.optimizers["bias"](self.bias,bias_grad,*arg))

