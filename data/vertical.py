import numpy

# Modified from:
# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def generate_vertical(samples=100, classes=3):
    X = numpy.zeros((samples * classes, 2))
    y = numpy.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        X[ix] = numpy.c_[numpy.random.randn(samples) * 0.1 + (class_number) / 3, 
                         numpy.random.randn(samples) * 0.1 + 0.5]
        y[ix] = class_number
    return X, y
