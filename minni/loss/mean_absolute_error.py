import numpy

from .loss import Loss


class MeanAbsoluteError(Loss):
    def forward(self, yhat, y):
        return numpy.mean(numpy.abs(y - yhat), axis=-1)
    
    def backward(self, yhat, y):
        n = len(yhat)       # number of samples
        N = len(yhat[0])    # number of outputs
        
        d_ell = numpy.mean(numpy.sign(y - yhat))
        return d_ell / N
