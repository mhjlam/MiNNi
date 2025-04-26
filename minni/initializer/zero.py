import numpy

from .initializer import Initializer


class Zero(Initializer):
    def weights(self, F_in, F_out):
        return numpy.zeros((1, F_out))

    def biases(self, F_out):
        return numpy.zeros((1, F_out))
