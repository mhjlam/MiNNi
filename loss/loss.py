import numpy
from abc import ABC, abstractmethod

class Loss(ABC):
    def compute(self, predictions, targets):
        return numpy.mean(self.forward(predictions, targets))
    
    @abstractmethod
    def forward(self, predictions, targets):
        raise NotImplementedError('Must override method \'forward\' in derived class')
    
    @abstractmethod
    def backward(self, predictions, targets):
        raise NotImplementedError('Must override method \'backward\' in derived class')
