import numpy

# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def generate(samples=100, classes=3, dimensionality=2):
    X = numpy.zeros((samples*classes, dimensionality))  # data matrix (each row = single example)
    y = numpy.zeros(samples*classes, dtype='uint8')     # class labels

    for j in range(classes):
        ix = range(samples*j, samples*(j+1))
        r = numpy.linspace(0.0, 1, samples)                                             # radius
        t = numpy.linspace(j*4, (j+1)*4, samples) + numpy.random.randn(samples) * 0.2   # theta
        
        X[ix] = numpy.c_[r*numpy.sin(t*2.5), r*numpy.cos(t*2.5)]
        y[ix] = j

    return X, y
