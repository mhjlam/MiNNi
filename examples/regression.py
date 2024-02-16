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
