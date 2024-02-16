from .layer import Layer

class Input(Layer):
    def forward(self, inputs, training):
        self.output = inputs

    def backward(self, dvalues):
        pass
