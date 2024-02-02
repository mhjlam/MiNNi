# Simple Neural Network

## Example Usage

```python
import snn
import snn.activation
import snn.data
import snn.layer
import snn.loss
import snn.optimizer

X, y = snn.data.generate_spiral(samples=100, classes=3)

dense1 = snn.layer.Dense(2, 64)
activation1 = snn.activation.ReLu()

dense2 = snn.layer.Dense(64, 3)
loss_activation = snn.activation.SoftmaxCrossEntropy()

optimizer = snn.optimizer.Adam(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):
    ### Forward pass ###
    loss = loss_activation.forward(dense2.forward(activation1.forward(dense1.forward(X))), y)
    accuracy = snn.loss.accuracy(loss_activation.output, y)
    
    if not epoch % 100:
        print(f'epoch: {epoch:>5}, ' +
              f'accuracy: {accuracy:.4f}, ' +
              f'loss: {loss:.4f}, ' +
              f'learning rate: {optimizer.current_learning_rate:.4f}')
    
    ### Backward pass ###
    dense1.backward(activation1.backward(dense2.backward(loss_activation.backward(loss_activation.output, y))))
    
    ### Optimizer ###
    optimizer.pre_update()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update()
```
