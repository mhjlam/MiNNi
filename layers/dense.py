from .layer import Layer

'''
This layer class is also called fully-connected or fc for short.
'''

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        Layer.__init__(self, n_inputs, n_neurons)

    def __str__(self, n=0):
        return Layer.__str__(self, n)

    def str(self, n=0):
        return Layer.str(self, n)

    def forward(self, inputs):
        return Layer.forward(self, inputs)
