import numpy
from .loss import Loss

class CrossEntropy(Loss):
    # predictions: softmax normalized probability outputs
    def forward(self, predictions, targets):
        # Clip the predictions
        predictions = numpy.clip(predictions, 1e-7, 1-1e-7)
        
        # Probabilities
        if len(targets.shape) == 1:     # Categorical labels
            confidences = predictions[range(len(predictions)), targets]
        elif len(targets.shape) == 2:   # One-hot encoded labels
            confidences = numpy.sum(predictions * targets, axis=1)
        
        # Losses
        return -numpy.log(confidences)

    def backward(self, predictions, targets):
        # If labels are sparse, turn them into a one-hot vector
        if len(targets.shape) == 1:
            targets = numpy.eye(len(predictions[0]))[targets]
        
        # Calculate gradient
        self.dinputs = (-targets / predictions) / len(predictions)
