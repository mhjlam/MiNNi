from abc import ABC, abstractmethod

class Activation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predictions(self, output):
        pass
