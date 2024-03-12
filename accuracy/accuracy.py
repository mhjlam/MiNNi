import numpy
from abc import ABC, abstractmethod

from ..mnn import Metric

def compare_regression(c, y):
    return numpy.absolute(c - y) < numpy.std(y) / 250

def compare_binary(c, y):
    return c == y

def compare_multiclass(c, y):
    if len(y.shape) == 2:
        y = numpy.argmax(y, axis=1)
    return c == y


class Accuracy(ABC):
    def __init__(self, compare_metric=Metric.BINARY):
        self.acc_sum = 0
        self.acc_len = 0
        
        match compare_metric:
            case Metric.REGRESSION:
                self.compare = compare_regression
            case Metric.BINARY:
                self.compare = compare_binary
            case Metric.MULTICLASS:
                self.compare = compare_multiclass
    
    def __call__(self, c, y):
        comparisons = self.compare(c, y)
        acc = numpy.mean(comparisons)
        
        self.acc_sum += numpy.sum(comparisons)
        self.acc_len += len(comparisons)
        
        return acc

    def avg(self):
        avg_acc = self.acc_sum / self.acc_len
        self.acc_sum = 0
        self.acc_len = 0
        return avg_acc

    def reset_avg(self):
        self.acc_sum = 0
        self.acc_len = 0
