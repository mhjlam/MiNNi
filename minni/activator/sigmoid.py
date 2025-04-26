import numpy
from .activator import Activator


class Sigmoid(Activator):
    def forward(self, x):
        self.z = 1 / (1 + numpy.exp(-x))
        return self.z

    def backward(self, dz):
        return dz * (1 - self.z) * self.z

    def predict(self, yhat):
        return (yhat > 0.5) * 1
