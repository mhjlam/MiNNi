import numpy
from .loss import Loss

class SoftmaxLoss(Loss):
    def forward(self, yhat, y): # forward of CCE loss
        yhat = numpy.clip(yhat, 1e-7, 1-1e-7)
        
        if len(y.shape) == 1: # Categorical labels
            yhat = yhat[range(len(yhat)), y]
        elif len(y.shape) == 2: # One-hot encoded labels
            yhat = numpy.sum(yhat * y, axis=1)
        
        return -numpy.log(yhat)
    
    def backward(self, yhat, y): # combined backward pass of CCE + Softmax
        n = len(yhat)
        
        if len(y.shape) == 2:
            y = numpy.argmax(y, axis=1)
        
        dL = yhat.copy()
        dL[range(n), y] -= 1
        return dL / n
