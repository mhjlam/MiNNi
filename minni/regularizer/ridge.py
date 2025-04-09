import numpy
from .regularizer import Regularizer

class Ridge(Regularizer): # L2 regularization
    def __init__(self, lambda_L2):
        self.lambda_L2 = lambda_L2
    
    def backward(self, theta):
        dtheta = numpy.zeros_like(theta)
        
        if self.lambda_L2 > 0:
            dtheta += 2 * self.lambda_L2 * theta
        
        return dtheta

    def loss(self, theta):
        loss = 0
        if self.lambda_L2 > 0:
            loss += self.lambda_L2 * numpy.sum(numpy.power(theta, 2))
        return loss
