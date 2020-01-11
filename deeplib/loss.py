"""
A loss function measures how good our predictions are.
We can use this to adjust the parameters of our network
"""
import numpy as np

from deeplib.tensor import Tensor

class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    MSE is mean squared error
    CrossEntropy for multiclass classification
    Add a regularization term to penalise for having larger weights
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum( (predicted - actual) ** 2 )
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)