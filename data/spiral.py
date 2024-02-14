import numpy

# Modified from:
# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def generate_spiral(samples, classes):
    X = numpy.zeros((samples*classes, 2))
    y = numpy.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = numpy.linspace(0.0, 1, samples)
        t = numpy.linspace(class_number*4, (class_number+1)*4, samples) + numpy.random.randn(samples)*0.2
        X[ix] = numpy.c_[r*numpy.sin(t*2.5), r*numpy.cos(t*2.5)]
        y[ix] = class_number
    return X, y
