from ..layer import Input
from ..activation import Softmax
from ..loss import CategoricalCrossEntropy
from ..activation import SoftmaxCrossEntropy

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
    
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss_func, optimizer, accuracy):
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Input()
        n_layers = len(self.layers)
        
        self.trainable_layers = []
        
        for i in range(n_layers):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < n_layers - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss_func
                self.output_layer_activation = self.layers[i]
        
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
            
        self.loss_func.keep_trainable_layers(self.trainable_layers)
        
        if isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss_func, CategoricalCrossEntropy):
            self.softmax_classifier_output = SoftmaxCrossEntropy()

    def train(self, X, y, *, epochs=1, print_freq=1, validation_data=None):
        self.accuracy.init(y)
        
        for epoch in range(1, epochs+1):
            # Forward pass
            output = self.forward(X, training=True)
            
            # Loss
            data_loss, reg_loss = self.loss_func.compute(output, y, include_regularization=True)
            loss = data_loss + reg_loss
            
            # Compute accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.compute(predictions, y)
            
            # Backward pass
            self.backward(output, y)
            
            # Optimize
            self.optimizer.pre_update()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update()
            
            if not epoch % print_freq:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f}, (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {reg_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
                
        if validation_data is not None:
            X_val, y_val = validation_data
            output = self.forward(X_val, training=False)
            loss = self.loss_func.compute(output, y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.compute(predictions, y_val)
            
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
            
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
        else:
            self.loss_func.backward(output, y)
            for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)
