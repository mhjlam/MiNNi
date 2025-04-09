import inspect
from abc import ABC, abstractmethod

from ..layer import Dense
from ..layer import Linear

class Optimizer(ABC):
    def __init__(self):
        pass
    
    def __call__(self, layers):
        self.optimize(layers)
    
    @abstractmethod
    def pre_update(self):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')
    
    @abstractmethod
    def update(self, layer):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')
    
    @abstractmethod
    def post_update(self):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')
    
    def optimize(self, layers):
        self.pre_update()
        for layer in layers:
            if isinstance(layer, (Dense, Linear)):
                self.update(layer)
        self.post_update()
