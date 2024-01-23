import numpy

from .layer import Layer
from ..snn import Gradient

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.type = 'Dense'
        self.weights = 0.01 * numpy.random.randn(n_inputs, n_neurons)
        self.biases = numpy.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        return numpy.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = numpy.dot(self.inputs.T, dvalues)
        self.dbiases = numpy.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = numpy.dot(dvalues, self.weights.T)
        
        return Gradient(self.dweights, self.dbiases, self.dinputs)
