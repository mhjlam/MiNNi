import numpy

from .layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size, stride, mode='max'):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride
        if mode not in ['max', 'avg']:
            raise ValueError("Invalid mode. Use 'max' for max pooling or 'avg' for average pooling.")
        self.mode = mode

    def forward(self, x):
        self.x = x
        batch_size, channels, height, width = x.shape
        pool_height, pool_width = self.pool_size

        # Calculate output dimensions
        out_height = (height - pool_height) // self.stride + 1
        out_width = (width - pool_width) // self.stride + 1

        # Initialize output tensor
        self.output = numpy.zeros((batch_size, channels, out_height, out_width))

        if self.mode == 'max':
            # Store indices of max values for backward pass
            self.max_indices = numpy.zeros_like(x, dtype=bool)

        # Perform pooling
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + pool_height
                w_start = j * self.stride
                w_end = w_start + pool_width

                x_slice = x[:, :, h_start:h_end, w_start:w_end]

                if self.mode == 'max':
                    # Max pooling
                    max_values = numpy.max(x_slice, axis=(2, 3), keepdims=True)
                    self.output[:, :, i, j] = max_values.squeeze()
                    # Store the mask of max values
                    self.max_indices[:, :, h_start:h_end, w_start:w_end] = (x_slice == max_values)
                elif self.mode == 'avg':
                    # Average pooling
                    avg_values = numpy.mean(x_slice, axis=(2, 3))
                    self.output[:, :, i, j] = avg_values

        return self.output

    def backward(self, dz):
        dx = numpy.zeros_like(self.x)

        out_height, out_width = dz.shape[2], dz.shape[3]
        pool_height, pool_width = self.pool_size

        # Distribute gradients
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + pool_height
                w_start = j * self.stride
                w_end = w_start + pool_width

                if self.mode == 'max':
                    # Backpropagate gradients for max pooling
                    dx[:, :, h_start:h_end, w_start:w_end] += dz[:, :, i, j][:, :, None, None] * self.max_indices[:, :, h_start:h_end, w_start:w_end]
                elif self.mode == 'avg':
                    # Backpropagate gradients for average pooling
                    avg_grad = dz[:, :, i, j][:, :, None, None] / (pool_height * pool_width)
                    dx[:, :, h_start:h_end, w_start:w_end] += avg_grad

        return dx
