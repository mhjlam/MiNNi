import numpy
from .initializer import Initializer

class Random(Initializer):
    def __init__(self, scaler=0.01):
        self.scaler = scaler
    
    def weights(self, F_in, F_out):
        numpy.random.seed(1337)
        return self.scaler * numpy.random.randn(F_in, F_out)

    def biases(self, F_in, F_out):
        return numpy.zeros((1, F_out))
