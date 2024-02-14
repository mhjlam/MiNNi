import numpy
from .optimizer import Optimizer

class Adam(Optimizer): # Adaptive Moment Estimation
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def pre_update(self):
        if self.decay: self.current_learning_rate = \
            self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = numpy.zeros_like(layer.weights)
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_momentums = numpy.zeros_like(layer.biases)
            layer.bias_cache = numpy.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = \
            self.beta_1 * layer.weight_momentums + (1.0 - self.beta_1) * layer.dweights
        layer.bias_momentums = \
            self.beta_1 * layer.bias_momentums + (1.0 - self.beta_1) * layer.dbiases

        # Calculate corrected momentum
        i = self.iterations + 1
        weight_momentums = layer.weight_momentums / (1.0 - self.beta_1**i)
        bias_momentums = layer.bias_momentums / (1.0 - self.beta_1**i)
        
        # Update cache with squared current gradients
        layer.weight_cache = \
            self.beta_2 * layer.weight_cache + (1.0 - self.beta_2) * layer.dweights**2
        layer.bias_cache = \
            self.beta_2 * layer.bias_cache + (1.0 - self.beta_2) * layer.dbiases**2

        # Calculate corrected cache
        weight_cache = layer.weight_cache / (1.0 - self.beta_2**i)
        bias_cache = layer.bias_cache / (1.0 - self.beta_2**i)

        # SGD parameter update with normalization weight square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums / \
                         (numpy.sqrt(weight_cache) + self.epsilon)
        layer.biases +=  -self.current_learning_rate * bias_momentums / \
                         (numpy.sqrt(bias_cache) + self.epsilon)
        
    def post_update(self):
        self.iterations += 1
