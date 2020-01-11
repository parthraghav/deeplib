"""
We usean optimizer to adjust the paramters of our network
basd on the gradients computed during backpropogation
"""

from deeplib.neural_net import NeuralNetwork

class Optimizer:
    def step(self, net: NeuralNetwork) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNetwork) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
            
