import numpy

from .loss import Loss


class BinaryCrossEntropy(Loss):
    def forward(self, yhat, y):
        yhat = numpy.clip(yhat, 1e-7, 1-1e-7)
        
        L = -(y * numpy.log(yhat) + (1 - y) * numpy.log(1 - yhat))
        L = numpy.mean(L, axis=-1)
        return L
    
    def backward(self, yhat, y):
        n = len(yhat)       # number of samples
        N = len(yhat[0])    # number of outputs
        yhat = numpy.clip(yhat, 1e-7, 1-1e-7)
        
        dL = -(y / yhat - (1 - y) / (1 - yhat))
        dL = dL / n
        return dL / N # normalize over outputs
