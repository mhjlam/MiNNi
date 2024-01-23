# Simple Neural Network

## Example Usage

```python
import snn
import snn.activation
import snn.data
import snn.layer
import snn.loss

samples, labels = snn.data.generate_spiral(100, 3)

dense1 = snn.layer.Dense(2, 3)
activation1 = snn.activation.ReLu()

dense2 = snn.layer.Dense(3, 3)
loss_activation = snn.activation.SoftmaxCrossEntropy()

def forward_pass(samples, labels):
    dense1_output = dense1.forward(samples)
    act1_output = activation1.forward(dense1_output)
    dense2_output = dense2.forward(act1_output)
    loss = loss_activation.forward(dense2_output, labels)
    
    print('\nFORWARD PASS')
    print(loss_activation.output[:5])
    print('loss:', loss)
    print('accuracy:', snn.loss.accuracy(loss_activation.output, labels))

def backward_pass():
    loss_activation_dinputs = loss_activation.backward(loss_activation.output, labels)
    dense2_gradients = dense2.backward(loss_activation_dinputs)
    activation1_dinputs = activation1.backward(dense2.dinputs)
    dense1_gradients = dense1.backward(activation1_dinputs)
    
    print('\nBACKWARD PASS')
    print(dense1_gradients.dweights)
    print(dense1_gradients.dbiases)
    print()
    print(dense2_gradients.dweights)
    print(dense2_gradients.dbiases)

forward_pass(samples, labels)
backward_pass()
```
