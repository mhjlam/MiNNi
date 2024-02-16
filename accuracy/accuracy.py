import numpy

class Accuracy:
    def compute(self, predictions, targets):
        comparisons = self.compare(predictions, targets)
        accuracy = numpy.mean(comparisons)
        return accuracy
