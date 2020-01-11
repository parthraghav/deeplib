"""
Here's a function that can train a nerual network
"""

from deeplib.tensor import Tensor
from deeplib.neural_net import NeuralNetwork
from deeplib.loss import Loss, MSE
from deeplib.optimiser import Optimizer, SGD
from deeplib.data import DataIterator, BatchIterator

def train(
    net: NeuralNetwork,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int = 5000,
    iterator: DataIterator = BatchIterator(),
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD()
    ) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            # take inputs out of the batch, and make a prediction
            predicted = net.forward(batch.inputs)
            # compute loss and add it to the epoch loss 
            epoch_loss += loss.loss(predicted, batch.targets)
            # compute the gradient
            grad = loss.grad(predicted, batch.targets)
            # and backpropogate with it
            net.backward(grad)

            optimizer.step(net)
        
        print(epoch, epoch_loss)


