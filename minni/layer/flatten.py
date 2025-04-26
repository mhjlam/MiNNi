from .layer import Layer


class Flatten(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dz):
        return dz.reshape(self.x_shape)
