import numpy
from .accuracy import Accuracy

class RegressionAccuracy(Accuracy):
    def __init__(self):
        self.precision = None
    
    def init(self, targets, reinit=False):
        if self.precision is None or reinit:
            self.precision = numpy.std(targets) / 250
    
    def compare(self, predictions, targets):
        return numpy.absolute(predictions - targets) < self.precision
