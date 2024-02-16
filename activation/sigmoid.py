import numpy
from .activation import Activation

class Sigmoid(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + numpy.exp(-inputs))
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs

    def predictions(self, output):
        return (output > 0.5) * 1
