import copy
import numpy
import pickle

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

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
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
            
        if self.loss is not None:
            self.loss.set_trainable_layers(self.trainable_layers)
        
        if isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = SoftmaxCrossEntropy()

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

    def train(self, X, y, *, epochs=1, batch_size=None, print_freq=1, validation_data=None):
        self.accuracy.init(y)

        #  Determine training steps based on batch_size
        train_steps = 1
        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            train_steps += 1 if train_steps * batch_size < len(X) else 0
        
        # Training
        for epoch in range(1, epochs+1):
            print_epoch = batch_size is None and not epoch % print_freq
            if batch_size is not None or print_epoch:
                print(f'epoch: {epoch}')
            
            # Reset loss and accuracy for this epoch
            self.loss.reset()
            self.accuracy.reset()
            
            # Iterate over batch steps
            for step in range(train_steps):
                batch_X = X
                batch_y = y

                # Slice a batch
                if batch_size is not None:
                    i0 = step*batch_size
                    i1 = (step+1)*batch_size
                    batch_X = X[i0:i1]
                    batch_y = y[i0:i1]
            
                # Forward pass
                output = self.forward(batch_X, training=True)
                
                # Compute loss
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
                print_batch = not step % print_freq or step == train_steps-1
                if batch_size is not None and print_batch:
                    print(f'batch: {step:>3},  ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {reg_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate:.20f}')
            
            # Epoch loss/accuracy
            epoch_data_loss, epoch_reg_loss = self.loss.accumulated(incl_reg=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.accumulated()
            
            if batch_size is not None or print_epoch:
                print(f'[training]   ' +
                    f'acc: {epoch_acc:.3f}, ' +
                    f'loss: {epoch_loss:.3f} (' +
                    f'data_loss: {epoch_data_loss:.3f}, ' +
                    f'reg_loss: {epoch_reg_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate:.20f}')
        
        # Validation
        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        self.loss.reset()
        self.accuracy.reset()
        
        for step in range(validation_steps):
            batch_X = X_val
            batch_y = y_val

            # Slice a batch
            if batch_size is not None:
                i0 = step*batch_size
                i1 = (step+1)*batch_size
                batch_X = X_val[i0:i1]
                batch_y = y_val[i0:i1]
            
            # Forward pass
            output = self.forward(batch_X, training=False)
            
            # Compute loss
            self.loss(output, batch_y)
            
            # Compute accuracy
            predictions = self.output_activation.predictions(output)
            self.accuracy(predictions, batch_y)
        
        # Validation loss/accuracy
        validation_loss = self.loss.accumulated()
        validation_acc = self.accuracy.accumulated()
        
        print(f'[validation] ' +
                f'acc: {validation_acc:.3f}, ' +
                f'loss: {validation_loss:.3f}')
    
    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        output = []
        for step in range(prediction_steps):
            batch_X = X
            if batch_size is not None:
                batch_X = X[step*batch_size:(step+1)*batch_size]
        
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
        
        # Stack arrays and return
        return numpy.vstack(output)

    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        for params, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*params)

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
        
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)
        
        # Reset accumulated values
        model.loss.reset()
        model.accuracy.reset()
        
        # Remove data from input layer and gradients from loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        # Remove layer inputs, output, and dinputs
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        # Save model to binary file
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
