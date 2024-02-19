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
        self.loss = loss_func
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
                self.layers[i].next = self.loss
                self.output_activation = self.layers[i]
        
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
            
        self.loss.set_trainable_layers(self.trainable_layers)
        
        if isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = SoftmaxCrossEntropy()

    def train(self, X, y, *, epochs=1, batch_size=None, print_freq=1, validation_data=None):
        self.accuracy.init(y)

        #  Determine training steps based on batch_size
        train_steps = 1
        
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            train_steps += 1 if train_steps * batch_size < len(X) else 0
            
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                validation_steps += 1 if validation_steps * batch_size < len(X_val) else 0
        
        # Training
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')
            
            # Reset loss and accuracy for this epoch
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            # Iterate over batch steps
            for step in range(train_steps):
                batch_X = X
                batch_y = y

                if batch_size is not None:
                    i0 = step*batch_size
                    i1 = (step+1)*batch_size
                    batch_X = X[i0:i1]
                    batch_y = y[i0:i1]
            
                # Forward pass
                output = self.forward(batch_X, training=True)
                
                # Loss
                data_loss, reg_loss = self.loss(output, batch_y, incl_reg=True)
                loss = data_loss + reg_loss
                
                # Compute accuracy
                predictions = self.output_activation.predictions(output)
                accuracy = self.accuracy(predictions, batch_y)
                
                # Backward pass
                self.backward(output, batch_y)
                
                # Optimize
                self.optimizer(self.trainable_layers)
                
                # Batch loss/accuracy
                if not step % print_freq or step == train_steps-1:
                    print(f'batch: {step:>3},  ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {reg_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate:.20f}')
            
            # Epoch loss/accuracy
            epoch_data_loss, epoch_reg_loss = \
                self.loss.accumulated(incl_reg=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.accumulated()
            
            print(f'[training]   ' +
                  f'acc: {epoch_acc:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_reg_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate:.20f}')
            
        # Validation
        if validation_data is not None:
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(validation_steps):
                batch_X = X_val
                batch_y = y_val

                if batch_size is not None:
                    i0 = step*batch_size
                    i1 = (step+1)*batch_size
                    batch_X = X_val[i0:i1]
                    batch_y = y_val[i0:i1]
                
                output = self.forward(batch_X, training=False)
                loss = self.loss(output, batch_y)
                predictions = self.output_activation.predictions(output)
                accuracy = self.accuracy(predictions, batch_y)
            
            # Validation loss/accuracy
            validation_loss = self.loss.accumulated()
            validation_acc = self.accuracy.accumulated()
            
            print(f'[validation] ' +
                  f'acc: {validation_acc:.3f}, ' +
                  f'loss: {validation_loss:.3f}')

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
            self.loss.backward(output, y)
            for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)
