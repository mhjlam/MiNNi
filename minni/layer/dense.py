import numpy

from .layer import Layer

from ..loss import CrossEntropy
from ..activator import Softmax
from ..activator import Rectifier
from ..regularizer import ElasticNet
from ..initializer import Random


class Dense(Layer):
    def __init__(self, F_in, F_out, initializer=Random(), activator=Rectifier(), regularizer=ElasticNet()):
        super().__init__(F_in, F_out, initializer, activator, regularizer)

    def forward(self, x):
        self.x = x

        # Forward pass of dense layer
        z = numpy.dot(x, self.W) + self.b

        # Forward pass of activator
        return self.activator.forward(z)

    def backward(self, dz, loss=None):
        if isinstance(self.activator, Softmax) and isinstance(loss, CrossEntropy):
            pass  # Skip activator when using CrossEntropy
        elif self.activator:
            # Backward pass of activator
            dz = self.activator.backward(dz)

        # Backward pass of dense layer
        self.dx = numpy.dot(dz, self.W.T)
        self.dW = numpy.dot(self.x.T, dz)
        self.db = numpy.sum(dz, axis=0, keepdims=True)

        # Regularization
        self.dW += self.regularizer(self.W)
        self.db += self.regularizer(self.b)

        return self.dx
