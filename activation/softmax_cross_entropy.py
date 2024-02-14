import numpy

from .activation import Activation
from .softmax import Softmax
from ..loss import CategoricalCrossEntropy

class SoftmaxCrossEntropy(Activation):
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
        
    def forward(self, inputs, targets):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.compute(self.output, targets)

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
