import numpy
from .activator import Activator


class Rectifier(Activator):
    def forward(self, x, train=False):
        self.x = x
        return numpy.maximum(0, x)

    def backward(self, dz):
        dx = dz.copy()
        dx[self.x <= 0] = 0
        return dx

    def predict(self, yhat):
        return yhat
