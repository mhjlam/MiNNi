import copy
import numpy
import pickle

from .. import Metric
from ..layer import Linear
from ..accuracy import Accuracy


class Model:
    def __init__(self, *, loss=None, optimizer=None, metric=Metric.MULTICLASS, verbose=True):
        self.layers = []
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = Accuracy(metric)
        self.verbose = verbose


    def add(self, layer):
        self.layers.append(layer)


    def forward(self, X):
        #print("Forward pass")
        prev_layer = Linear()
        z = prev_layer.forward(X)
        for layer in self.layers:
            #print(f"Layer type: {type(layer).__name__}")
            z = layer.forward(z)
            prev_layer = layer
        c = prev_layer.activator.predict(z)
        return z, c


    def backward(self, yhat, y):
        #print("Backward pass")
        dx = self.loss.backward(yhat, y)
        dx = self.layers[-1].backward(dx, self.loss)
        for layer in reversed(self.layers[:-1]):
            #print(f"Layer type: {type(layer).__name__}")
            dx = layer.backward(dx)
        return dx


    def train(self, X, y, epochs=1, batch_size=None):
        # Determine batch size and number of steps
        batch_size = batch_size or len(X)
        steps = -(len(X) // -batch_size)

        num_digits_epochs = len(str(epochs))
        num_digits_steps = len(str(steps))

        for epoch in range(1, epochs + 1):
            for t in range(steps):
                # Batch slice
                batch_X = X[t * batch_size : (t + 1) * batch_size]
                batch_y = y[t * batch_size : (t + 1) * batch_size]

                # Forward pass
                yhat, c = self.forward(batch_X)

                # Loss
                opt_loss = self.loss.data_loss(yhat, batch_y)
                reg_loss = self.loss.reg_loss(self.layers)
                loss = opt_loss + reg_loss

                # Accuracy
                acc = self.accuracy(c, batch_y)

                # Backward pass
                self.backward(yhat, batch_y)

                # Optimize
                self.optimizer.optimize(self.layers)

                # Show batch statistics every 10% of the time
                if self.verbose and batch_size != len(X) and (t + 1 == steps or t % (steps // 10) == 0):
                    print(f"[Batch: {t:>{num_digits_steps}}] "
                        + f"\n\tAccuracy: {acc:.3f} "
                        + f"\n\tLoss: {loss:.3f} ("
                        + f"Optimizer: {opt_loss:.3f}, "
                        + f"Regularizer: {reg_loss:.3f}) "
                        + f"\n\tLearning rate: {self.optimizer.eta_t:.8f}")

            # Epoch statistics
            epoch_opt_loss = self.loss.avg()
            epoch_reg_loss = self.loss.reg_loss(self.layers)
            epoch_loss = epoch_opt_loss + epoch_reg_loss
            epoch_acc = self.accuracy.avg()

            # Print epoch statistics every 10% of the time
            if epoch % epochs < 2 or epoch % (epochs // 10) == 0:
                print(f"[Epoch {epoch:>{num_digits_epochs}}]"
                    + f"\n\tLoss: {epoch_loss:.3f} ("
                    + f"Optimizer: {epoch_opt_loss:.3f}, "
                    + f"Regularizer: {epoch_reg_loss:.3f}) "
                    + f"\n\tAccuracy: {epoch_acc:.3f} "
                    + f"\n\tLearning rate: {self.optimizer.eta_t:.8f}")


    def evaluate(self, X, y, batch_size=None):
        # Remove Dropout layers
        self.layers = [layer for layer in self.layers if layer.__class__.__name__ != "Dropout"]
        
        # Determine batch size and number of steps
        batch_size = batch_size or len(X)
        steps = -(len(X) // -batch_size)

        for t in range(steps):
            # Batch slice
            batch_X = X[t * batch_size : (t + 1) * batch_size]
            batch_y = y[t * batch_size : (t + 1) * batch_size]

            # Forward pass
            yhat, c = self.forward(batch_X)

            # Batch loss
            self.loss.data_loss(yhat, batch_y)

            # Batch accuracy
            self.accuracy(c, batch_y)

        # Validation loss/accuracy
        val_loss = self.loss.avg()
        val_acc = self.accuracy.avg()

        print(
            f"[Validation] "
            + f"\n\tLoss: {val_loss:.3f}, "
            + f"\n\tAccuracy: {val_acc:.3f}"
        )


    def predict(self, X, batch_size=None):
        # Remove Dropout layers
        self.layers = [layer for layer in self.layers if layer.__class__.__name__ != "Dropout"]

        # Determine batch and step size
        batch_size = batch_size or len(X)
        steps = -(len(X) // -batch_size)

        yhat = []
        for t in range(steps):
            batch_X = X[t * batch_size : (t + 1) * batch_size]
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

        with open(path, "wb") as f:
            pickle.dump(model, f)


    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
