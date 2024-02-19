import numpy
from abc import ABC, abstractmethod

class Loss(ABC):
    def __call__(self, predictions, targets, incl_reg=False):
        return self.loss(predictions, targets, incl_reg)
    
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
                    numpy.sum(layer.biases * layer.biases)

        return regularization_loss
        
    def set_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def loss(self, predictions, targets, incl_reg=False):
        sample_losses = self.forward(predictions, targets)
        data_loss = numpy.mean(sample_losses)
        
        self.accumulated_sum += numpy.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        if not incl_reg:
            return data_loss
        
        return data_loss, self.regularization_loss()

    def accumulated(self, *, incl_reg=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not incl_reg:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    @abstractmethod
    def forward(self, predictions, targets):
        raise NotImplementedError('Must override method \'forward\' in derived class')
    
    @abstractmethod
    def backward(self, predictions, targets):
        raise NotImplementedError('Must override method \'backward\' in derived class')
