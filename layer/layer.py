import abc
import numpy
import inspect

from ..initializer import Zero

class Layer(abc.ABC):
    def __init__(self, F_in, F_out, initializer=Zero(), activator=None, regularizer=None):
        self.W = initializer.weights(F_in, F_out)
        self.b = initializer.biases(F_in, F_out)
        
        self.mW = numpy.zeros_like(self.W)
        self.mb = numpy.zeros_like(self.b)
        self.vW = numpy.zeros_like(self.W)
        self.vb = numpy.zeros_like(self.b)
        
        self.activator = activator
        self.regularizer = regularizer
    
    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')

    @abc.abstractmethod
    def backward(self, dz):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')
