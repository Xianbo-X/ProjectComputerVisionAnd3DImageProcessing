from NeuralNetworkProj.Optim.LearningRate import LearningRate

class NaiveUpdator():
    def __init__(self,lr:LearningRate) -> None:
        self.learning_rate=lr

    def __call__(self, variable,grad):
        return self.naive_updator(variable,grad)

    def naive_updator(self,variable,grad):
        return variable-self.learning_rate.get_lr()*grad