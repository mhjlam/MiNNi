import numpy
from .layer import Layer

class MaxPooling(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x, train):
        self.x = x
        out_height = (x.shape[1] - self.pool_size) // self.stride + 1
        out_width = (x.shape[2] - self.pool_size) // self.stride + 1
        out = numpy.zeros((x.shape[0], out_height, out_width, x.shape[3]))
        
        for i in range(out_height):
            for j in range(out_width):
                x_slice = x[:, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, :]
                out[:, i, j, :] = numpy.max(x_slice, axis=(1, 2))
        
        return out
    
    def backward(self, dz):
        pass
