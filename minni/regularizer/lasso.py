import numpy
from .regularizer import Regularizer

class Lasso(Regularizer): # L1 regularization
    def __init__(self, lambda_L1):
        self.lambda_L1 = lambda_L1
    
    def backward(self, theta):
        dtheta = numpy.zeros_like(theta)
        
        if self.lambda_L1 > 0:
            dL1 = numpy.ones_like(theta)
            dL1[theta < 0] = -1
            dtheta += self.lambda_L1 * dL1

        return dtheta

    def loss(self, theta):
        loss = 0
        if self.lambda_L1 > 0:
            loss += self.lambda_L1 * numpy.sum(numpy.abs(theta))
        return loss
