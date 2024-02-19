from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__():
        pass
    
    def __call__(self, layers):
        self.update(layers)
    
    @abstractmethod
    def pre_update(self):
        pass
    
    @abstractmethod
    def update_params(self, layer):
        pass

    @abstractmethod
    def post_update(self):
        pass

    def update(self, layers):
        self.pre_update()
        for layer in layers:
            self.update_params(layer)
        self.post_update()
