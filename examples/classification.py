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
