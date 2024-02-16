import numpy
from abc import ABC, abstractmethod

class Loss(ABC):
    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0: # L1 regularization (weights)
                regularization_loss += layer.weight_regularizer_l1 * \
                    numpy.sum(numpy.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0: # L2 regularization (weights)
                regularization_loss += layer.weight_regularizer_l2 * \
                    numpy.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:   # L1 regularization (biases)
                regularization_loss += layer.bias_regularizer_l1 * \
                    numpy.sum(numpy.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:   # L2 regularization (biases)
                regularization_loss += layer.bias_regularizer_l2 * \
                    numpy.sum(layer.biases*layer.biases)

        return regularization_loss
        
    def keep_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def compute(self, predictions, targets, *, include_regularization=False):
        sample_losses = self.forward(predictions, targets)
        data_loss = numpy.mean(sample_losses)
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    @abstractmethod
    def forward(self, predictions, targets):
        raise NotImplementedError('Must override method \'forward\' in derived class')
    
    @abstractmethod
    def backward(self, predictions, targets):
        raise NotImplementedError('Must override method \'backward\' in derived class')
