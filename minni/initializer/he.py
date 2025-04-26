import numpy

from .initializer import Initializer


class He(Initializer):
    def __init__(self):
        pass

    def weights(self, F_in, F_out):
        stddev = numpy.sqrt(2.0 / F_in)
        return numpy.random.randn(F_in, F_out) * stddev

    def biases(self, F_out):
        return numpy.zeros(F_out)
