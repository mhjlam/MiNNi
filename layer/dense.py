import numpy

from .layer import Layer

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, weight_scale=0.01,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = weight_scale * numpy.random.randn(n_inputs, n_neurons)
        self.biases = numpy.zeros((1, n_neurons))
        
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = numpy.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = numpy.dot(self.inputs.T, dvalues)
        self.dbiases = numpy.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = numpy.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.bias_regularizer_l1 > 0:
            dL1 = numpy.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = numpy.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
