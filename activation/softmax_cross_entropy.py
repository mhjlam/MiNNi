import numpy

from .activation import Activation
from .softmax import Softmax
from ..loss import CrossEntropy

class SoftmaxCrossEntropy(Activation):
    def __init__(self):
        self.activation = Softmax()
        self.loss = CrossEntropy()
        
    def forward(self, inputs, targets):
        self.output = self.activation.forward(inputs)
        return self.loss.compute(self.output, targets), self.output

    def backward(self, predictions, targets):
        # Convert one-hot encoded labels into discrete values
        if len(targets.shape) == 2:
            targets = numpy.argmax(targets, axis=1)
        
        # Compute gradient
        n_samples = len(predictions)
        self.dinputs = predictions.copy()
        self.dinputs[range(n_samples), targets] -= 1        
        self.dinputs = self.dinputs / n_samples
        return self.dinputs
