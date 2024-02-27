import numpy
from .loss import Loss

class MeanSquaredError(Loss): # L2 loss
    def forward(self, predictions, targets):
        sample_losses = numpy.mean((targets - predictions)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, targets):
        n_samples = len(dvalues)
        n_outputs = len(dvalues[0])
        
        self.dinputs = -2 * (targets - dvalues)
        self.dinputs = self.dinputs / n_outputs
        self.dinputs = self.dinputs / n_samples
        return self.dinputs
