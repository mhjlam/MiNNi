import numpy
from .layer import Layer

'''
This layer class is also called fully-connected or fc for short.
'''

class Dense(Layer):
    def __init__(self, inputs, n_neurons, activation=None):
        self.type = 'Dense'
        self.inputs = inputs
        self.weights = 0.01 * numpy.random.randn(numpy.shape(inputs)[1], n_neurons)
        self.biases = numpy.zeros((1, n_neurons))
        self.activation = activation
   
    def __str__(self, n=0):
        n = 1 if n < 1 else n
        return f'Layer {str(n)}:\n  type: {self.type}\n\n' + \
               f'  Inputs:\n' + \
               f'    {str(self.inputs)}' + \
               f'  Weights:\n' + \
               f'    {str(self.weights)}' + \
               f'  Biases:\n' + \
               f'    {str(self.biases)}'
             
    def str(self, n=0):
        return Layer.str(self, n)

    def forward(self):
        if self.activation:
            return self.activation(numpy.dot(self.inputs, self.weights) + self.biases)
        else:
            return numpy.dot(self.inputs, self.weights) + self.biases
