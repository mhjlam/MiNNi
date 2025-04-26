import inspect
from abc import ABC, abstractmethod


class Regularizer(ABC):
    @abstractmethod
    def __init__(self, lambda_term):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')
    
    @abstractmethod
    def backward(self, theta):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')

    @abstractmethod
    def loss(self, theta):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')

    def __call__(self, theta):
        return self.backward(theta)
