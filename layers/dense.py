import numpy
from .layer import Layer

'''
This layer class is also called fully-connected or fc for short.
'''

class Dense(Layer):
    def __init__(self, inputs, n_neurons):
      self.type = 'Dense'
      self.inputs = inputs
      self.weights = 0.01 * numpy.random.randn(numpy.shape(inputs)[1], n_neurons)
      self.biases = numpy.zeros((1, n_neurons))
   
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
        return Layer.forward(self, self.inputs)
