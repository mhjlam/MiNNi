import numpy
from abc import ABC, abstractmethod

class Loss(ABC):
    def compute(self, predictions, targets):
        return numpy.mean(self.forward(predictions, targets))

    def regularization_loss(self, layer):
        regularization_loss = 0
        
        if layer.regularizer.l1w > 0:
            regularization_loss += layer.regularizer.l1w * numpy.sum(numpy.abs(layer.weights))
        
        if layer.regularizer.l2w > 0:
            regularization_loss += layer.regularizer.l2w * numpy.sum(layer.weights * layer.weights)
        
        if layer.regularizer.l1b > 0:
            regularization_loss += layer.regularizer.l1b * numpy.sum(numpy.abs(layer.biases))
        
        if layer.regularizer.l2b > 0:
            regularization_loss += layer.regularizer.l2b * numpy.sum(layer.biases * layer.biases)

        return regularization_loss
        
    @abstractmethod
    def forward(self, predictions, targets):
        raise NotImplementedError('Must override method \'forward\' in derived class')
    
    @abstractmethod
    def backward(self, predictions, targets):
        raise NotImplementedError('Must override method \'backward\' in derived class')
