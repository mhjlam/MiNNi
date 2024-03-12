import copy
import numpy
import pickle

from ..layer import Linear
from ..layer import Dropout
from ..accuracy import Accuracy

from ..mnn import Metric

class Model():
    def __init__(self, *, loss=None, optimizer=None, metric=Metric.MULTICLASS):
        self.layers = []
        
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = Accuracy(metric)
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X, train=False):
        prev_layer = Linear()
        z = prev_layer.forward(X, train)
        for layer in self.layers:
            z = layer.forward(z, train)
            prev_layer = layer
        c = prev_layer.activator.predict(z)        
        return z, c
    
    def backward(self, yhat, y):
        dx = self.loss.backward(yhat, y)
        dx = self.layers[-1].backward(dx, self.loss)
        for layer in reversed(self.layers[:-1]):
            dx = layer.backward(dx)
        return dx
    
    def train(self, X, y, epochs=1, batch_size=None):
        # Determine batch size and number of steps
        batch_size = batch_size or len(X)
        steps = -(len(X) // -batch_size)
        
        num_digits_epochs = len(str(epochs))
        num_digits_steps = len(str(steps))
        
        for e in range(1, epochs+1):
            for t in range(steps):
                # Batch slice
                batch_X = X[t*batch_size:(t+1)*batch_size]
                batch_y = y[t*batch_size:(t+1)*batch_size]
                
                # Forward pass
                yhat, c = self.forward(batch_X, True)
                
                # Loss
                data_loss = self.loss.data_loss(yhat, batch_y)
                reg_loss = self.loss.reg_loss(self.layers)
                loss = data_loss + reg_loss
                
                # Accuracy
                acc = self.accuracy(c, batch_y)
                
                # Backward pass
                self.backward(yhat, batch_y)
                
                # Optimize
                self.optimizer.optimize(self.layers)
                
                # Show batch statistics every 10% of the time
                if batch_size != len(X) and (t+1 == steps or t % (steps // 10) == 0):
                    print(f'[batch: {t:>{num_digits_steps}}] ' +
                        f'acc: {acc:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {reg_loss:.3f}), ' +
                        f'lr: {self.optimizer.eta_t:.20f}')
            
            # Epoch statistics
            epoch_data_loss = self.loss.avg()
            epoch_reg_loss = self.loss.reg_loss(self.layers)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.avg()

            # Print epoch statistics every 10% of the time
            if e % epochs < 2 or e % (epochs // 10) == 0:
                print(f'[epoch {e:>{num_digits_epochs}}] ' +
                      f'loss: {epoch_loss:.3f}, (' +
                      f'data_loss: {epoch_data_loss:.3f}, ' +
                      f'reg_loss: {epoch_reg_loss:.3f}), ' +
                      f'acc: {epoch_acc:.3f}, ' +
                      f'eta: {self.optimizer.eta_t:.20f}')
    
    def evaluate(self, X, y, batch_size=None):
        # Determine batch size and number of steps
        batch_size = batch_size or len(X)
        steps = -(len(X) // -batch_size)
        
        for t in range(steps):
            # Batch slice
            batch_X = X[t*batch_size:(t+1)*batch_size]
            batch_y = y[t*batch_size:(t+1)*batch_size]
            
            # Forward pass
            yhat, c = self.forward(batch_X)
            
            # Batch loss
            self.loss.data_loss(yhat, batch_y)
            
            # Batch accuracy
            self.accuracy(c, batch_y)
        
        # Validation loss/accuracy
        val_loss = self.loss.avg()
        val_acc = self.accuracy.avg()
        
        print(f'[validation] ' +
              f'loss: {val_loss:.3f}, ' +
              f'acc: {val_acc:.3f}')
    
    def predict(self, X, batch_size=None):
        # Determine batch and step size
        batch_size = batch_size or len(X)
        steps = -(len(X) // -batch_size)
        
        yhat = []
        for t in range(steps):
            batch_X = X[t*batch_size:(t+1)*batch_size]
            batch_yhat = self.forward(batch_X)[0]
            yhat.append(batch_yhat)
        
        yhat = numpy.vstack(yhat)
        yhat = self.layers[-1].activator.predict(yhat)
        return yhat
    
    def save(self, path):
        model = copy.deepcopy(self)
        
        self.loss.reset_avg()
        self.accuracy.reset_avg()
        
        # Remove all properties except weights, biases, and regularization terms
        for layer in model.layers:
            layer.__dict__.pop("x", None)
            layer.__dict__.pop("z", None)
            layer.__dict__.pop("dx", None)
            layer.__dict__.pop("dW", None)
            layer.__dict__.pop("db", None)
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
