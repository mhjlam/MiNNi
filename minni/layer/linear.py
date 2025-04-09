from .layer import Layer

class Linear(Layer):
    def __init__(self):
        pass
    
    def forward(self, x, train):
        return x

    def backward(self, dz):
        return dz
