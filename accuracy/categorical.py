import numpy
from .accuracy import Accuracy

class Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary
    
    def init(self, targets):
        pass
    
    def compare(self, predictions, targets):
        if not self.binary and len(targets.shape) == 2:
            targets = numpy.argmax(targets, axis=1)
        return predictions == targets
