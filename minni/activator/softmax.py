import numpy
from .activator import Activator

class Softmax(Activator):
    def forward(self, x):
        logits = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
        self.z = logits / numpy.sum(logits, axis=1, keepdims=True)
        return self.z
    
    def backward(self, dz):
        dx = numpy.empty_like(dz)
        for j, (z_j, dz_j) in enumerate(zip(self.z, dz)):
            z_j = z_j.reshape(-1, 1)
            J = numpy.diagflat(z_j) - numpy.dot(z_j, z_j.T)
            dx[j] = numpy.dot(J, dz_j)
        return dx
    
    def predict(self, yhat):
        return numpy.argmax(yhat, axis=1)
