class Neuron:
    def __init__(self, inputs, weights, bias):
        if len(inputs) is not len(weights):
            raise ValueError('number of inputs and weights must be equal')
        self.inputs = inputs or []
        self.weights = weights or []
        self.biases = bias or 0

    def __str__(self, n=-1):
        return 'Neuron{}\n  inputs: {}\n  weights: {}\n  biases: {}\n'.format(
            f' {str(n)}:' if n >= 0 else ':', 
            str(self.inputs), 
            str(self.weights), 
            str(self.biases))

    def str(self, n=-1):
        try:
            return self.__str__(n)
        except:
            return self.__str__()
