import numpy
from .layer import Layer

class Flatten(Layer):
    def forward(self, x, train):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dz):
        return dz.reshape(self.x_shape)
