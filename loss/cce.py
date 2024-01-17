import numpy

def categorical_cross_entropy(predictions, targets):
    samples = len(predictions) # number of samples in batch
    
    # clip the predictions
    predictions = numpy.clip(predictions, 1e-7, 1-1e-7)
    
    # probabilities
    if len(targets.shape) == 1:
        confidences = predictions[range(samples), targets]
    elif len(targets.shape) == 2:
        confidences = numpy.sum(predictions * targets, axis=1)
    
    # losses
    return numpy.mean(-numpy.log(confidences)) # mean loss
