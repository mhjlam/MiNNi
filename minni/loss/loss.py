import numpy
import inspect
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self):
        self.acc_sum = 0
        self.acc_len = 0
    
    def __call__(self, yhat, y, layers):
        return self.data_loss(yhat, y), self.reg_loss(layers)
    
    def data_loss(self, yhat, y):
        sample_loss = self.forward(yhat, y)
        data_loss = numpy.mean(sample_loss)
        
        self.acc_sum += numpy.sum(sample_loss)
        self.acc_len += len(sample_loss)
        
        return data_loss
    
    def reg_loss(self, layers):
        reg_loss = 0
        
        for layer in layers:
            if hasattr(layer, 'regularizer') and layer.regularizer:
                reg_loss += layer.regularizer.loss(layer.W)
                reg_loss += layer.regularizer.loss(layer.b)
        
        return reg_loss
    
    def avg(self):
        avg_loss = self.acc_sum / self.acc_len
        self.acc_sum = 0
        self.acc_len = 0
        return avg_loss
    
    def reset_avg(self):
        self.acc_sum = 0
        self.acc_len = 0        
    
    @abstractmethod
    def forward(self, yhat, y):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')

    @abstractmethod    
    def backward(self, yhat, y):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')
