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
