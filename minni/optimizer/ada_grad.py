import numpy

from .optimizer import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, eta=1.0, beta=0.0):
        self.eta = eta
        self.beta = beta
        
        self.t = 0
        self.eta_t = eta
        self.epsilon = 1e-7
    
    def pre_update(self):
        self.eta_t = self.eta * (1.0 / (1.0 + self.beta * self.t))
    
    def update(self, layer):
        layer.vW += numpy.power(layer.dW, 2)
        layer.vb += numpy.power(layer.db, 2)
        
        layer.W += -(self.eta_t / (numpy.sqrt(layer.vW) + self.epsilon)) * layer.dW
        layer.b += -(self.eta_t / (numpy.sqrt(layer.vb) + self.epsilon)) * layer.db
    
    def post_update(self):
        self.t += 1
