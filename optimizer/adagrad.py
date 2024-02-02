import numpy
from .optimizer import Optimizer

class AdaGrad(Optimizer): # Adaptive Gradient
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = \
                self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_cache = numpy.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / \
                         (numpy.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases +=  -self.current_learning_rate * layer.dbiases / \
                         (numpy.sqrt(layer.bias_cache) + self.epsilon)
        
    def post_update(self):
        self.iterations += 1
