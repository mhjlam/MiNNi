import numpy

def Softmax(inputs):
    # Unnormalized probabilities
    probabilities = numpy.exp(inputs - numpy.max(inputs, axis=1, keepdims=True))
    
    # Return normalized probabilities for each sample
    return probabilities / numpy.sum(probabilities, axis=1, keepdims=True)
