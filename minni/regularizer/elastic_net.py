import numpy

from .regularizer import Regularizer


class ElasticNet(Regularizer): # L1 + L2 regularization
    def __init__(self, lambda_L1=0.0, lambda_L2=0.0):
        self.lambda_L1 = lambda_L1
        self.lambda_L2 = lambda_L2
    
    def backward(self, theta):
        dtheta = numpy.zeros_like(theta)
        
        if self.lambda_L1 > 0.0:
            dL1 = numpy.ones_like(theta)
            dL1[theta < 0.0] = -1.0
            dtheta += self.lambda_L1 * dL1
        
        if self.lambda_L2 > 0.0:
            dtheta += 2.0 * self.lambda_L2 * theta

        return dtheta

    def loss(self, theta):
        loss = 0.0
        if self.lambda_L1 > 0.0:
            loss += self.lambda_L1 * numpy.sum(numpy.abs(theta))
        if self.lambda_L2 > 0.0:
            loss += self.lambda_L2 * numpy.sum(numpy.power(theta, 2.0))
        return loss
