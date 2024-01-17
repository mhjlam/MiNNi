import numpy

def accuracy(predictions, targets):
    # Calculate accuracy from softmax output and targets
    predictions = numpy.argmax(predictions, axis=1)
    targets = numpy.argmax(targets, axis=1) if len(targets.shape) == 2 else targets
    return numpy.mean(predictions == targets)
