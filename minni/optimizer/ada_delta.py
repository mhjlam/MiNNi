import numpy

from .optimizer import Optimizer


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.accumulated_gradients = {}
        self.accumulated_updates = {}

    def pre_update(self):
        pass

    def update(self, layer):
        if layer not in self.accumulated_gradients:
            # Initialize accumulators for gradients and updates
            self.accumulated_gradients[layer] = {
                "W": numpy.zeros_like(layer.W),
                "b": numpy.zeros_like(layer.b)
            }
            self.accumulated_updates[layer] = {
                "W": numpy.zeros_like(layer.W),
                "b": numpy.zeros_like(layer.b)
            }

        # Update weights
        self.accumulated_gradients[layer]["W"] = (
            self.rho * self.accumulated_gradients[layer]["W"] +
            (1 - self.rho) * numpy.square(layer.dW))
        
        update_W = (
            numpy.sqrt(self.accumulated_updates[layer]["W"] + self.epsilon) /
            numpy.sqrt(self.accumulated_gradients[layer]["W"] + self.epsilon)) * layer.dW
        
        self.accumulated_updates[layer]["W"] = (
            self.rho * self.accumulated_updates[layer]["W"] +
            (1 - self.rho) * numpy.square(update_W))
        
        layer.W -= update_W

        # Update biases
        self.accumulated_gradients[layer]["b"] = (
            self.rho * self.accumulated_gradients[layer]["b"] +
            (1 - self.rho) * numpy.square(layer.db))
        
        update_b = (
            numpy.sqrt(self.accumulated_updates[layer]["b"] + self.epsilon) /
            numpy.sqrt(self.accumulated_gradients[layer]["b"] + self.epsilon)) * layer.db
        
        self.accumulated_updates[layer]["b"] = (
            self.rho * self.accumulated_updates[layer]["b"] +
            (1 - self.rho) * numpy.square(update_b))
        
        layer.b -= update_b

    def post_update(self):
        pass
