import numpy
from .optimizer import Optimizer

class GradientDescent(Optimizer):
    # eta = learning rate
    # beta = decay rate
    # gamma = momentum
    def __init__(self, eta=1.0, beta=0.0, gamma=0.0):
        self.eta = eta
        self.beta = beta
        self.gamma = gamma
        
        self.t = 0
        self.eta_t = eta
    
    def pre_update(self):
        self.eta_t = self.eta * (1.0 / (1.0 + self.beta * self.t))
    
    def update(self, layer):
        if self.gamma:
            layer.vW = self.gamma * layer.vW - self.eta_t * layer.dW
            layer.vb = self.gamma * layer.vb - self.eta_t * layer.db
            
            layer.W += layer.vW
            layer.b += layer.vb
        else:
            layer.W += -self.eta * layer.dW
            layer.b += -self.eta * layer.db
    
    def post_update(self):
        self.t += 1
