import numpy
from .optimizer import Optimizer

class RMSProp(Optimizer):
    def __init__(self, eta=0.001, beta=0.0, rho=0.9):
        self.eta = eta
        self.beta = beta
        
        self.t = 0
        self.eta_t = eta
        self.epsilon = 1e-7
    
    def pre_update(self):
        self.eta_t = self.eta * (1.0 / (1.0 + self.beta * self.t))
    
    def update(self, layer):
        layer.vW = self.rho * layer.vW + (1 - self.rho) * numpy.power(layer.dW, 2)
        layer.vb = self.rho * layer.vb + (1 - self.rho) * numpy.power(layer.db, 2)
        
        layer.W += -(self.eta_t / (numpy.sqrt(layer.vW) + self.epsilon)) * layer.dW
        layer.b += -(self.eta_t / (numpy.sqrt(layer.vb) + self.epsilon)) * layer.db
    
    def post_update(self):
        self.t += 1
