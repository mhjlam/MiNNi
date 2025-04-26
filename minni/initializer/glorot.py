import numpy

from .initializer import Initializer


class Glorot(Initializer):
    def __init__(self):
        pass

    def weights(self, F_in, F_out):
        limit = numpy.sqrt(6 / (F_in + F_out))
        return numpy.random.uniform(-limit, limit, (F_in, F_out))

    def biases(self, F_out):
        return numpy.zeros(F_out)
