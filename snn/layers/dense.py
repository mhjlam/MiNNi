import snn.neurons

class Dense:
    def __init__(self, neurons):
        if type(neurons) is not list:
            raise TypeError('expected a list of neuron objects')
        if not isinstance(neurons[0], snn.neurons.Neuron):
            raise TypeError('expected a list of neuron objects')
        self.neurons = neurons

    def __str__(self, n=0):
        label = f'Layer {str(n)}:\n' if n >= 0 else 'Layer:\n'
        return label + '  type: Dense\n\n' + \
            '\n'.join(['  ' + '  '.join(n.str(i).splitlines(True)) 
                       for i, n in enumerate(self.neurons)])

    def str(self, n=0):
        try:
            return self.__str__(n)
        except:
            return self.__str__()
