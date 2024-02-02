from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__():
        pass
    
    @abstractmethod
    def pre_update(self):
        pass
    
    @abstractmethod
    def update_params(self, layer):
        pass

    @abstractmethod
    def post_update(self):
        pass
