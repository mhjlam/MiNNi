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
relu1 = snn.activation.ReLu()

dense2 = snn.layer.Dense(64, 3)
sce_loss = snn.activation.SoftmaxCrossEntropy()

#optimizer = snn.optimizer.Momentum(decay=1e-3, momentum=0.9)
#optimizer = snn.optimizer.AdaGrad(decay=1e-4)
#optimizer = snn.optimizer.RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = snn.optimizer.Adam(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):
    ### Forward pass ###
    loss, out = sce_loss.forward(dense2.forward(relu1.forward(dense1.forward(X))), y)
    accuracy = snn.loss.accuracy(out, y)

    ### Progress ###
    if not epoch % 100:
        print(snn.epoch_stats(epoch, accuracy, loss, optimizer.current_learning_rate))
    
    ### Backward pass ###
    dense1.backward(relu1.backward(dense2.backward(sce_loss.backward(out, y))))
    
    ### Optimizer ###
    optimizer.update([dense1, dense2])
```
