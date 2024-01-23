import numpy
from .activation import Activation

class ReLu(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        return numpy.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
        # Zero gradient where values were negative
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
