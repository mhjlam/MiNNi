import numpy
import inspect
from abc import ABC, abstractmethod

class Accuracy(ABC):
    def __init__(self):
        self.acc_sum = 0
        self.acc_len = 0
    
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

    @abstractmethod
    def compare(self, c, y):
        raise NotImplementedError(f'Must override method \'{inspect.stack()[0][3]}\' in derived class')


class Regression(Accuracy):
    def compare(self, c, y):
        return numpy.absolute(c - y) < numpy.std(y) / 250

class BinaryCategorical(Accuracy):
    def compare(self, c, y):
        return c == y

class Categorical(Accuracy):
    def compare(self, c, y):
        if len(y.shape) == 2:
            y = numpy.argmax(y, axis=1)
        return c == y
