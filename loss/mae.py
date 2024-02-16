import numpy
from .loss import Loss

class MeanAbsoluteError(Loss): # L1 loss
    def forward(self, predictions, targets):
        sample_losses = numpy.mean(numpy.abs(targets - predictions), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, targets):
        n_samples = len(dvalues)
        n_outputs = len(dvalues[0])
        
        self.dinputs = numpy.sign(targets - dvalues)
        self.dinputs = self.dinputs / n_outputs
        self.dinputs = self.dinputs / n_samples
        return self.dinputs
