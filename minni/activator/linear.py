from .activator import Activator

class Linear(Activator):
    def forward(self, x):
        return x.copy()
    
    def backward(self, dz):
        return dz.copy()
    
    def predict(self, yhat):
        return yhat
