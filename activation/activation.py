from abc import ABC, abstractmethod

class Activation(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError('Must override method \'forward\' in derived class')
    
    def backward(self, dvalues):
        raise NotImplementedError('Must override method \'backward\' in derived class')
