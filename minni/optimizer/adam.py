import numpy

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, eta=0.001, beta=0.0, beta_1=0.9, beta_2=0.999):
        self.eta = eta
        self.beta = beta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        self.t = 0
        self.eta_t = eta
        self.epsilon = 1e-7
    
    def pre_update(self):
        self.eta_t = self.eta * (1.0 / (1.0 + self.beta * self.t))
    
    def update(self, layer):
        layer.mW = self.beta_1 * layer.mW + (1.0 - self.beta_1) * layer.dW
        layer.mb = self.beta_1 * layer.mb + (1.0 - self.beta_1) * layer.db
        
        mW_hat = layer.mW / (1.0 - numpy.power(self.beta_1, self.t+1))
        mb_hat = layer.mb / (1.0 - numpy.power(self.beta_1, self.t+1))
        
        layer.vW = self.beta_2 * layer.vW + (1.0 - self.beta_2) * numpy.power(layer.dW, 2)
        layer.vb = self.beta_2 * layer.vb + (1.0 - self.beta_2) * numpy.power(layer.db, 2)
        
        vW_hat = layer.vW / (1.0 - numpy.power(self.beta_2, self.t+1))
        vb_hat = layer.vb / (1.0 - numpy.power(self.beta_2, self.t+1))
        
        layer.W += -self.eta_t * mW_hat / (numpy.sqrt(vW_hat) + self.epsilon)
        layer.b += -self.eta_t * mb_hat / (numpy.sqrt(vb_hat) + self.epsilon)
    
    def post_update(self):
        self.t += 1
