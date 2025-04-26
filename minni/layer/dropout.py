import numpy

from .layer import Layer
from ..activator import Linear


class Dropout(Layer):
    def __init__(self, p):
        self.q = 1 - p
        
        self.activator = Linear()
        self.regularizer = None
    
    def forward(self, x):
        self.mask = numpy.random.binomial(1, self.q, size=x.shape) / self.q
        return x * self.mask
    
    def backward(self, dz):
        return dz * self.mask
