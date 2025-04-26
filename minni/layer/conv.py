import numpy
import scipy.signal

from .layer import Layer

from ..activator import Rectifier
from ..initializer import Random


class Conv(Layer):
    def __init__(self, input_nodes, output_nodes, kernel_size=3, stride=1, padding=0, initializer=Random(), activator=Rectifier()):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.activator = activator

        # Initialize weights and biases
        F_in = input_nodes * self.kernel_size[0] * self.kernel_size[1]
        F_out = output_nodes
        self.W = initializer.weights(F_out, F_in).reshape(output_nodes, input_nodes, *self.kernel_size)
        self.b = initializer.biases(F_out).reshape(output_nodes, 1)

    def forward(self, x):
        # Ensure input is in (batch_size, channels, height, width) format
        if x.ndim == 4 and x.shape[-1] == self.input_nodes:
            x = numpy.transpose(x, (0, 3, 1, 2))

        self.x = x
        batch_size, channels, height, width = x.shape

        # Apply padding
        x_padded = numpy.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        # Perform convolution using scipy's convolve
        z = numpy.zeros((batch_size, self.output_nodes, height, width))
        for b in range(batch_size):
            for o in range(self.output_nodes):
                for c in range(channels):
                    z[b, o] += scipy.signal.convolve(x_padded[b, c], self.W[o, c], mode='valid')
                z[b, o] += self.b[o]

        # Apply activation function
        return self.activator.forward(z)

    def backward(self, dz):
        batch_size, _, _, _ = dz.shape
        _, channels, height, width = self.x.shape

        # Initialize gradients
        self.dW = numpy.zeros_like(self.W)
        self.db = numpy.zeros_like(self.b)
        dx_padded = numpy.zeros((batch_size, channels, height + 2 * self.padding, width + 2 * self.padding))

        # Compute gradients using scipy's correlate
        for b in range(batch_size):
            for o in range(self.output_nodes):
                for c in range(channels):
                    # Gradient of weights
                    self.dW[o, c] += scipy.signal.correlate(self.x[b, c], dz[b, o], mode='valid')
                    # Gradient of input
                    dx_padded[b, c] += scipy.signal.convolve(dz[b, o], self.W[o, c], mode='full')
                # Gradient of biases
                self.db[o] += dz[b, o].sum()

        # Remove padding from dx
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        return dx
