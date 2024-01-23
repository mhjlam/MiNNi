import numpy

# Calculate accuracy from softmax output and targets
def accuracy(predictions, targets):
    predictions = numpy.argmax(predictions, axis=1)
    if len(targets.shape) == 2:
        targets = numpy.argmax(targets, axis=1)
    return numpy.mean(predictions == targets)
