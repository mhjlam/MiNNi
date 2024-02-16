import numpy

from .layer import Layer

class Dropout(Layer):
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return self.output
        
        self.binary_mask = numpy.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
