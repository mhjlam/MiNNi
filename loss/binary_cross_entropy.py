import numpy
from .loss import Loss

class BinaryCrossEntropy(Loss):
    def forward(self, predictions, targets):
        # Clip predictions
        pred = numpy.clip(predictions, 1e-7, 1-1e-7)
        
        sample_losses = -(targets * numpy.log(pred) + (1 - targets) * numpy.log(1 - pred))
        sample_losses = numpy.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, targets):
        n_samples = len(dvalues)
        n_outputs = len(dvalues[0])
        
        # Clip dvalues
        dvalues = numpy.clip(dvalues, 1e-7, 1-1e-7)
        
        self.dinputs = -(targets / dvalues - (1 - targets) / (1 - dvalues))
        self.dinputs = self.dinputs / n_outputs
        self.dinputs = self.dinputs / n_samples
        return self.dinputs
