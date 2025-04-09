import inspect
from abc import ABC, abstractmethod

class Initializer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def weights(self, F_in, F_out):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')

    @abstractmethod
    def biases(self, F_in, F_out):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')
