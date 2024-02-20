import numpy

class Accuracy:
    def __call__(self, predictions, targets):
        return self.accuracy(predictions, targets)
    
    def accuracy(self, predictions, targets):
        comparisons = self.compare(predictions, targets)
        accuracy = numpy.mean(comparisons)
        
        self.accumulated_sum += numpy.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy

    def accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    def reset(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
