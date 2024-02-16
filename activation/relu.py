import numpy
from .activation import Activation

class ReLU(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = numpy.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
        # Zero gradient where values were negative
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

    def predictions(self, output):
        return output
