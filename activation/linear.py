from .activation import Activation

class Linear(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy() # derivative: 1 * dvalues (chain rule)
        return self.dinputs
