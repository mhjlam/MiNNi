import numpy
from .activation import Activation

class Softmax(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        
        # Normalized probabilities for each sample
        exp_values = numpy.exp(inputs - numpy.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / numpy.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = numpy.empty_like(dvalues)
        
        # Iterate sample-wise over pairs of the outputs and gradients
        for index, (single_output, dvalues_set) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Compute Jacobian of the output (matrix of all partial derivatives)
            jacobian = numpy.diagflat(single_output) - numpy.dot(single_output, single_output.T)
            
            # Compute sample-wise gradient and add to array of sample gradients
            self.dinputs[index] = numpy.dot(jacobian, dvalues_set)
        return self.dinputs

    def predictions(self, output):
        return numpy.argmax(output, axis=1)
