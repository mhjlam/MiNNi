import numpy

from .activation import Activation

class SoftmaxCrossEntropy(Activation):
    def backward(self, dvalues, targets):
        n_samples = len(dvalues)
        
        # Convert one-hot encoded labels into discrete values
        if len(targets.shape) == 2:
            targets = numpy.argmax(targets, axis=1)
        
        # Compute gradient
        self.dinputs = dvalues.copy()
        self.dinputs[range(n_samples), targets] -= 1
        self.dinputs = self.dinputs / n_samples
        return self.dinputs

    def predictions(self, output):
        pass
