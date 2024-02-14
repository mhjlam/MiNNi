import numpy
from abc import ABC, abstractmethod

class Loss(ABC):
    def compute(self, predictions, targets):
        sample_losses = self.forward(predictions, targets)
        data_loss = numpy.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):
        regularization_loss = 0

        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                numpy.sum(numpy.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                numpy.sum(layer.weights * layer.weights)

        # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                numpy.sum(numpy.abs(layer.biases))

        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                numpy.sum(layer.biases*layer.biases)

        return regularization_loss
        
    @abstractmethod
    def forward(self, predictions, targets):
        raise NotImplementedError('Must override method \'forward\' in derived class')
    
    @abstractmethod
    def backward(self, predictions, targets):
        raise NotImplementedError('Must override method \'backward\' in derived class')
