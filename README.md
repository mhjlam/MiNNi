# Simple Neural Network

## Example Usage

```python
import snn
import snn.activation
import snn.dataset
import snn.layers
import snn.loss

X, y = snn.dataset.generate_spiral(samples=100, classes=3)

# Use Rectified Linear activation function on the hidden layer
layer1 = snn.layers.Dense(inputs=X, n_neurons=3, activation=snn.activation.ReLU)

# Use Softmax activation function for the output layer
layer2 = snn.layers.Dense(inputs=layer1.forward(), n_neurons=3, activation=snn.activation.Softmax)

y_hat = layer2.forward()
print(y_hat[:5])

loss = snn.loss.categorical_cross_entropy(y_hat, y)
print('loss:', loss)

accuracy = snn.loss.accuracy(y_hat, y)
print('accuracy:', accuracy)
```
