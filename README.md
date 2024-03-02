# Simple Neural Network

## Example Usage

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

model.set(loss=snn.loss.MeanSquaredError(), 
          optimizer=snn.optimizer.Adam(learning_rate=0.005, decay=1e-3),
          accuracy=snn.accuracy.Regression())

model.finalize()

model.train(X, y, epochs=10000, print_freq=100)
```

### Classification (Spiral Data)

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

model.set(loss=snn.loss.CategoricalCrossEntropy(), 
          optimizer=snn.optimizer.Adam(learning_rate=0.05, decay=5e-5),
          accuracy=snn.accuracy.Categorical())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_freq=100)
```

### MNIST Fashion Dataset Classification

#### Training

```python
import os

import snn
import snn.loss
import snn.layer
import snn.model
import snn.assets
import snn.accuracy
import snn.optimizer
import snn.activation

mnist_fashion_images_dir = \
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\assets\\fashion_mnist_images'))
snn.assets.import_dataset_mnist(mnist_fashion_images_dir)

X, y = snn.assets.load_dataset_mnist('train', mnist_fashion_images_dir)
X_test, y_test = snn.assets.load_dataset_mnist('test', mnist_fashion_images_dir)

X, y = snn.assets.preprocess_dataset_mnist(X, y)
X_test = snn.assets.preprocess_test_dataset_mnist(X_test)

model = snn.model.Model()
model.add(snn.layer.Dense(X.shape[1], 128))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(128, 128))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(128, 10))
model.add(snn.activation.Softmax())

model.set(
    loss=snn.loss.CategoricalCrossEntropy(),
    optimizer=snn.optimizer.Adam(decay=1e-3),
    accuracy=snn.accuracy.Categorical()
)
model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_freq=100)

model.evaluate(X, y)

model.save(os.path.join(os.path.dirname(__file__), 'fashion_mnist.snnm'))
```

#### Testing

```python
import os

import snn.model
import snn.assets

asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\assets'))
fashion_mnist_images_dir = os.path.join(asset_dir, 'fashion_mnist_images')

snn.assets.import_dataset_mnist(fashion_mnist_images_dir)

X_test, y_test = snn.assets.load_dataset_mnist('test', fashion_mnist_images_dir)

X_test = snn.assets.preprocess_test_dataset_mnist(X_test)

model = snn.model.Model.load(os.path.join(os.path.dirname(__file__), 'fashion_mnist.snnm'))

model.evaluate(X_test, y_test)

confidences = model.predict(X_test[:5])
predictions = model.output_activation.predictions(confidences)

for prediction in predictions:
    print(prediction, snn.assets.FASHION_MNIST_LABELS[prediction])
```

#### Prediction

```python
import os
import cv2
import numpy

import snn.model
import snn.assets

example_dir = os.path.join(os.path.dirname(__file__))
asset_dir = os.path.abspath(os.path.join(example_dir, '..\\assets'))

model = snn.model.Model.load(os.path.join(example_dir, 'fashion_mnist.snnm'))

def predict_image(image_filename):
    image_data = cv2.imread(os.path.join(asset_dir, image_filename), cv2.IMREAD_GRAYSCALE)
    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    image_data = (image_data.reshape(1,-1).astype(numpy.float32) - 127.5) / 127.5

    confidences = model.predict(image_data)
    predictions = model.output_activation.predictions(confidences)
    print(image_filename + '\t' + snn.assets.FASHION_MNIST_LABELS[predictions[0]])

predict_image('tshirt.png')
predict_image('pants.png')
```
