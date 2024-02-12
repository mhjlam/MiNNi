# Simple Neural Network

## Example Usage

```python
import enum

import snn
import snn.activation
import snn.data
import snn.layer
import snn.loss
import snn.optimizer

Optimizer = enum.Enum('Optimizer', ['MOMENTUM', 'ADAGRAD', 'RMSPROP', 'ADAM'])

use_dropout = False
use_optimizer = Optimizer.ADAM

X, y = snn.data.generate_spiral(samples=1000, classes=3)
X_test, y_test = snn.data.generate_spiral(samples=100, classes=3)

dense1 = snn.layer.Dense(2, 512, snn.Regularizer(0, 0, 5e-4, 5e-4))
dense2 = snn.layer.Dense(512, 3)

if use_dropout:
    dropout1 = snn.layer.Dropout(0.1)
    
activation1 = snn.activation.ReLu()
loss_activation = snn.activation.SoftmaxCrossEntropy()

if use_optimizer == Optimizer.MOMENTUM:
    optimizer = snn.optimizer.Momentum(decay=1e-3, momentum=0.9)
elif use_optimizer == Optimizer.ADAGRAD:
    snn.optimizer.AdaGrad(decay=1e-4)
elif use_optimizer == Optimizer.RMSPROP:
    snn.optimizer.RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
else:
    optimizer = snn.optimizer.Adam(learning_rate=0.02, decay=5e-7)

def train(X, y):
    for epoch in range(10001):
        # Forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        
        if use_dropout:
            dropout1.forward(activation1.output)
            dense2.forward(dropout1.output)
        else:        
            dense2.forward(activation1.output)
        
        data_loss, loss_out = loss_activation.forward(dense2.output, y)
        regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                            loss_activation.loss.regularization_loss(dense2)
        loss = data_loss + regularization_loss
        accuracy = snn.accuracy(loss_out, y)
        
        snn.show_epoch_stats(epoch, accuracy, loss, data_loss, regularization_loss, 
                             optimizer.current_learning_rate)
        
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        
        if use_dropout:
            dropout1.backward(dense2.dinputs)
            activation1.backward(dropout1.dinputs)
        else:
            activation1.backward(dense2.dinputs)
            
        dense1.backward(activation1.dinputs)
        
        # Optimize
        optimizer.pre_update()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update()
    
def validate(X_test, y_test):
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss, loss_out = loss_activation.forward(dense2.output, y_test)
    accuracy = snn.accuracy(loss_out, y_test)
    print(f'validation acc: {accuracy:.3f}, loss: {loss:.3f}')

if __name__== '__main__':
    train(X, y)
    validate(X_test, y_test)
```
