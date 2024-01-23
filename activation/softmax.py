import numpy
from .activation import Activation

class Softmax(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        
        # Unnormalized probabilities
        exp_values = numpy.exp(inputs - numpy.max(inputs, axis=1, keepdims=True))
        
        # Return normalized probabilities for each sample
        self.output = exp_values / numpy.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = numpy.empty_like(dvalues)
        
        # Iterate sample-wise over pairs of the outputs and gradients
        for i, (output, gradiant_set) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            output = output.reshape(-1, 1)
            
            # Compute Jacobian of the output (matrix of all partial derivatives)
            jacobian = numpy.diagflat(output) - numpy.dot(output, output.T)
            
            # Compute sample-wise gradient and add to array of sample gradients
            self.dinputs[i] = numpy.dot(jacobian, gradiant_set)
