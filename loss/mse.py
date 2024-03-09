import numpy
from .loss import Loss

class MeanSquaredError(Loss):
    def forward(self, yhat, y):
        return numpy.mean(numpy.power(y - yhat, 2), axis=-1)
    
    def backward(self, yhat, y):
        n = len(yhat)       # number of samples
        N = len(yhat[0])    # number of outputs
        
        d_ell = -(2/n) * (y - yhat)
        return d_ell / N
