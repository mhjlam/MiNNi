# Simple Neural Network

## Example Usage

### Classification

```python
import snn
import snn.data
import snn.loss
import snn.layer
import snn.model
import snn.accuracy
import snn.optimizer
import snn.activation

X, y = snn.data.generate_spiral(samples=1000, classes=3)
X_test, y_test = snn.data.generate_spiral(samples=100, classes=3)

model = snn.model.Model()
model.add(snn.layer.Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dropout(0.1))
model.add(snn.layer.Dense(512, 3))
model.add(snn.activation.Softmax())

model.set(loss_func=snn.loss.CategoricalCrossEntropy(), 
          optimizer=snn.optimizer.Adam(learning_rate=0.05, decay=5e-5),
          accuracy=snn.accuracy.CategoricalAccuracy())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_freq=100)
```

### Regression

```python
import snn
import snn.data
import snn.loss
import snn.layer
import snn.model
import snn.accuracy
import snn.optimizer
import snn.activation

X, y = snn.data.generate_sine(samples=1000)

model = snn.model.Model()
model.add(snn.layer.Dense(1, 64, weight_scale=0.1))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(64, 64, weight_scale=0.1))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(64, 1, weight_scale=0.1))
model.add(snn.activation.Linear())

model.set(loss_func=snn.loss.MeanSquaredError(), 
          optimizer=snn.optimizer.Adam(learning_rate=0.005, decay=1e-3),
          accuracy=snn.accuracy.RegressionAccuracy())

model.finalize()

model.train(X, y, epochs=10000, print_freq=100)
```
