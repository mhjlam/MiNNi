import copy
import snn.neurons
import snn.layers

neuron0 = snn.neurons.Neuron([1,2,3], [0.2,0.8,-0.5], 2)
neuron1 = copy.deepcopy(neuron0)

layer = snn.layers.Dense([neuron0, neuron1])

print(layer)
