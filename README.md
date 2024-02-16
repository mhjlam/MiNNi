# Simple Neural Network

## Example Usage

### Classification

```python
import enum
import numpy

import snn
import snn.activation
import snn.data
import snn.layer
import snn.loss
import snn.optimizer

Optimizer = enum.Enum('Optimizer', ['MOMENTUM', 'ADAGRAD', 'RMSPROP', 'ADAM'])

use_dropout = False
use_optimizer = Optimizer.ADAM

def train(model, X, y):
    for epoch in range(10001):
        # Forward pass
        model.dense1.forward(X)
        model.activation1.forward(model.dense1.output)
        
        if use_dropout:
            model.dropout1.forward(model.activation1.output)
            model.dense2.forward(model.dropout1.output)
        else:
            model.dense2.forward(model.activation1.output)
        
        data_loss = model.loss_activation.forward(model.dense2.output, y)
        regularization_loss = model.loss_activation.loss.regularization_loss(model.dense1) + \
                              model.loss_activation.loss.regularization_loss(model.dense2)
        loss = data_loss + regularization_loss
        
        predictions = numpy.argmax(model.loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = numpy.argmax(y, axis=1)
        accuracy = numpy.mean(predictions==y)
        
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, '+
                  f'loss: {loss:.3f} (' +
                  f'data_loss: {data_loss:.3f}, ' +
                  f'reg_loss: {regularization_loss:.3f}), ' +
                  f'lr: {model.optimizer.current_learning_rate}')
        
        # Backward pass
        model.loss_activation.backward(model.loss_activation.output, y)
        model.dense2.backward(model.loss_activation.dinputs)
        
        if use_dropout:
            model.dropout1.backward(model.dense2.dinputs)
            model.activation1.backward(model.dropout1.dinputs)
        else:
            model.activation1.backward(model.dense2.dinputs)
        
        model.dense1.backward(model.activation1.dinputs)
        
        # Optimize
        model.optimizer.pre_update()
        model.optimizer.update_params(model.dense1)
        model.optimizer.update_params(model.dense2)
        model.optimizer.post_update()
    
def validate(model, X_test, y_test):
    model.dense1.forward(X_test)
    model.activation1.forward(model.dense1.output)
    model.dense2.forward(model.activation1.output)

    loss = model.loss_activation.forward(model.dense2.output, y_test)
    
    predictions = numpy.argmax(model.loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = numpy.argmax(y_test, axis=1)
    accuracy = numpy.mean(predictions == y_test)    
    print(f'validation acc: {accuracy:.3f}, loss: {loss:.3f}')

class Model:
    def __init__(self):
        self.dense1 = snn.layer.Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
        self.dense2 = snn.layer.Dense(512, 3)
        self.dropout1 = snn.layer.Dropout(0.1) if use_dropout else None
        
        self.activation1 = snn.activation.ReLU()
        self.loss_activation = snn.activation.SoftmaxCrossEntropy()
        
        if use_optimizer == Optimizer.MOMENTUM:
            self.optimizer = snn.optimizer.Momentum(decay=1e-3, momentum=0.9)
        elif use_optimizer == Optimizer.ADAGRAD:
            self.optimizer = snn.optimizer.AdaGrad(decay=1e-4)
        elif use_optimizer == Optimizer.RMSPROP:
            self.optimizer = snn.optimizer.RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
        elif use_optimizer == Optimizer.ADAM or use_optimizer == None:
            self.optimizer = snn.optimizer.Adam(learning_rate=0.02, decay=5e-7)

if __name__== '__main__':
    X, y = snn.data.generate_spiral(samples=1000, classes=3)
    
    model = Model()
    train(model, X, y)
    
    X_test, y_test = snn.data.generate_spiral(samples=100, classes=3)
    validate(model, X_test, y_test)
```

### Binary Logistic Regression

```python
import numpy

import snn
import snn.activation
import snn.data
import snn.layer
import snn.loss
import snn.optimizer

def train(model, X, y):
    for epoch in range(10001):
        # Forward pass
        model.dense1.forward(X)
        model.activation1.forward(model.dense1.output)
        model.dense2.forward(model.activation1.output)
        model.activation2.forward(model.dense2.output)
        
        # Loss
        data_loss = model.loss_function.compute(model.activation2.output, y)
        regularization_loss = model.loss_function.regularization_loss(model.dense1) + \
                              model.loss_function.regularization_loss(model.dense2)
        loss = data_loss + regularization_loss
        
        # Accuracy
        predictions = (model.activation2.output > 0.5) * 1
        accuracy = numpy.mean(predictions == y)
        
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, '+
                  f'loss: {loss:.3f} (' +
                  f'data_loss: {data_loss:.3f}, ' +
                  f'reg_loss: {regularization_loss:.3f}), ' +
                  f'lr: {model.optimizer.current_learning_rate}')
        
        # Backward pass
        model.loss_function.backward(model.activation2.output, y)
        model.activation2.backward(model.loss_function.dinputs)
        model.dense2.backward(model.activation2.dinputs)
        model.activation1.backward(model.dense2.dinputs)
        model.dense1.backward(model.activation1.dinputs)
        
        # Optimize
        model.optimizer.pre_update()
        model.optimizer.update_params(model.dense1)
        model.optimizer.update_params(model.dense2)
        model.optimizer.post_update()
    
def validate(model, X_test, y_test):
    model.dense1.forward(X_test)
    model.activation1.forward(model.dense1.output)
    model.dense2.forward(model.activation1.output)
    model.activation2.forward(model.dense2.output)
    
    # Loss, accuracy
    data_loss = model.loss_function.compute(model.activation2.output, y_test)
    predictions = (model.activation2.output > 0.5) * 1
    accuracy = numpy.mean(predictions == y_test)

    print(f'validation acc: {accuracy:.3f}, loss: {data_loss:.3f}')

class Model:
    def __init__(self):
        self.dense1 = snn.layer.Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
        self.dense2 = snn.layer.Dense(64, 1)

        self.activation1 = snn.activation.ReLU()
        self.activation2 = snn.activation.Sigmoid()

        self.loss_function = snn.loss.BinaryCrossEntropy()
        self.optimizer = snn.optimizer.Adam(decay=5e-7)
        
if __name__== '__main__':
    X, y = snn.data.generate_spiral(samples=100, classes=2)
    y = y.reshape(-1, 1)
    
    model = Model()
    train(model, X, y)
    
    X_test, y_test = snn.data.generate_spiral(samples=100, classes=2)
    y_test = y_test.reshape(-1, 1)
    validate(model, X_test, y_test)
```

### Regression

```python
import numpy
import matplotlib.pyplot

import snn
import snn.activation
import snn.data
import snn.layer
import snn.loss
import snn.optimizer

X, y = snn.data.generate_sine(samples=1000)

dense1 = snn.layer.Dense(1, 64, weight_scale=0.1)
dense2 = snn.layer.Dense(64, 64, weight_scale=0.1)
dense3 = snn.layer.Dense(64, 1, weight_scale=0.1)

activation1 = snn.activation.ReLU()
activation2 = snn.activation.ReLU()
activation3 = snn.activation.Linear()

loss_func = snn.loss.MeanSquaredError()
optimizer = snn.optimizer.Adam(learning_rate=0.005, decay=1e-3)

accuracy_precision = numpy.std(y) / 250

for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    data_loss = loss_func.compute(activation3.output, y)
    regularization_loss = loss_func.regularization_loss(dense1) + \
                          loss_func.regularization_loss(dense2) + \
                          loss_func.regularization_loss(dense3)
    loss = data_loss + regularization_loss
    
    accuracy = numpy.mean(numpy.absolute(activation3.output - y) < accuracy_precision)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')
    
    # Backward pass
    loss_func.backward(activation3.output, y)
    activation3.backward(loss_func.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Optimize
    optimizer.pre_update()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update()

# Testing
X_test, y_test = snn.data.generate_sine(samples=1000)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

matplotlib.pyplot.plot(X_test, y_test)
matplotlib.pyplot.plot(X_test, activation3.output)
matplotlib.pyplot.show()
```