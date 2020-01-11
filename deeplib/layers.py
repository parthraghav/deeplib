"""
Our neural networks will be made up of layers.
Each layer needs to pass its inputs forward
and propogate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable

import numpy as np

from deeplib.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the output corresponding to these inputs
        """
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropogate this gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Computes output = inputs @ w + b
    @ is the matrix multiplication
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs * w + b
        """
        # save a copy of inputs because we are going to use it when we backpropogate
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        # we want to return the gradient with respect to input
        return grad @ self.params['w'].T

F = Callable[[Tensor], Tensor]
class ActivationLayer(Layer):
    """
    An activation layer just applies a function elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x  = g(z)
        then dy/dz = g'(z) * f'(x)
        """
        return self.f_prime(self.inputs) * grad



def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__(tanh, tanh_prime)