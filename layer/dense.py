import numpy

from .layer import Layer
from ..snn import Regularizer

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, regularizer=Regularizer(0,0,0,0)):
        self.weights = 0.01 * numpy.random.randn(n_inputs, n_neurons)
        self.biases = numpy.zeros((1, n_neurons))
        self.regularizer = regularizer

    def forward(self, inputs):
        self.inputs = inputs
        self.output = numpy.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = numpy.dot(self.inputs.T, dvalues)
        self.dbiases = numpy.sum(dvalues, axis=0, keepdims=True)
        
        if self.regularizer.l1w > 0:
            dL1 = numpy.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.regularizer.l1w * dL1
        if self.regularizer.l2w > 0:
            self.dweights += 2 * self.regularizer.l2w * self.weights
        if self.regularizer.l1b > 0:
            dL1 = numpy.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.regularizer.l1b * dL1
        if self.regularizer.l2b > 0:
            self.dbiases += 2 * self.regularizer.l2b * self.biases
        
        self.dinputs = numpy.dot(dvalues, self.weights.T)        
        return self.dinputs
