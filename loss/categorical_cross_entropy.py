import numpy
from .loss import Loss

class CategoricalCrossEntropy(Loss):
    # predictions: softmax normalized probability outputs
    def forward(self, predictions, targets):
        n_samples = len(predictions)
        
        # Clip the predictions
        predictions = numpy.clip(predictions, 1e-7, 1-1e-7)
        
        # Probabilities
        if len(targets.shape) == 1:     # Categorical labels
            confidences = predictions[range(n_samples), targets]
        
        elif len(targets.shape) == 2:   # One-hot encoded labels
            confidences = numpy.sum(predictions * targets, axis=1)
        
        # Losses
        return -numpy.log(confidences)

    def backward(self, dvalues, targets):
        n_samples = len(dvalues)
        n_labels = len(dvalues[0])
        
        # If labels are sparse, turn them into a one-hot vector
        if len(targets.shape) == 1:
            targets = numpy.eye(n_labels)[targets]
        
        # Calculate gradient
        self.dinputs = -targets / dvalues
        self.dinputs = self.dinputs / n_samples
        return self.dinputs
