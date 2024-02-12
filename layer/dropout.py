import numpy

from .layer import Layer

class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.rate = dropout_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = numpy.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
