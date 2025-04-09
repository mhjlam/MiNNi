import numpy
from .layer import Layer

class Conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initializer=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        self.W = initializer.weights((out_channels, in_channels, kernel_size, kernel_size))
        self.b = initializer.biases((out_channels, 1))

    def forward(self, x, train):
        self.x = x
        # Add padding
        x_padded = numpy.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        
        # Output dimensions
        out_height = (x.shape[1] - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (x.shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = numpy.zeros((x.shape[0], out_height, out_width, self.out_channels))
        
        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                x_slice = x_padded[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]
                out[:, i, j, :] = numpy.tensordot(x_slice, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b.T
        
        return out
    
    def backward(self, dz):
        pass
