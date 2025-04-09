import numpy
from .loss import Loss

class CategoricalCrossEntropy(Loss):
    def forward(self, yhat, y):
        yhat = numpy.clip(yhat, 1e-7, 1-1e-7)
        return -numpy.log(numpy.sum(yhat * y, axis=1))
    
    def backward(self, yhat, y):
        n = len(yhat)
        N = len(yhat[0])
        
        dL = -(1/n) * (y / yhat)
        return dL / N
