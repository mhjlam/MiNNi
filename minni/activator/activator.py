import inspect
from abc import ABC, abstractmethod


class Activator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError(
            f"Must override method '{inspect.stack()[0][3]}' in derived class"
        )

    @abstractmethod
    def backward(self, dz):
        raise NotImplementedError(
            f"Must override method '{inspect.stack()[0][3]}' in derived class"
        )

    @abstractmethod
    def predict(self, yhat):
        raise NotImplementedError(
            f"Must override method '{inspect.stack()[0][3]}' in derived class"
        )
