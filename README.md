# Mini Neural Network

MiNNi (Mini Neural Network) is a lightweight, educational neural network framework built from scratch in Python. Designed with simplicity and clarity in mind, MiNNi provides a clean, intuitive API for building, training, and deploying neural networks without the complexity of larger frameworks.

The implementation is based on the book [Neural Networks from Scratch](https://nnfs.io) by Harrison Kinsley & Daniel Kukiela.

## Key Features

- **Pure Python Implementation**: Built from the ground, using only Python 3 and NumPy.
- **Modular Architecture**: Clean separation of concerns with dedicated modules for layers, activators, optimizers, loss functions, and more.
- **Multiple Problem Types**: Supports regression, binary classification, and multiclass classification tasks.
- **Flexible Model Building**: Easy-to-use API for constructing neural network models with various layer types and configurations.
- **Built-in Regularization**: Includes dropout layers and regularization techniques to prevent overfitting.
- **Model Persistence**: Save and load trained models for later use.
- **Educational Focus**: Designed to be readable and understandable for learning purposes.

## Example Usage

### Regression

```python
model = minni.model.Model(
    loss=minni.loss.MeanSquaredError(),
    optimizer=minni.optimizer.Adam(eta=0.005, beta=1e-3),
    accuracy=minni.accuracy.Regression())

rand_scaled = minni.initializer.Random(scaler=0.1)
model.add(minni.layer.Dense(1, 64, rand_scaled, minni.activator.Rectifier()))
model.add(minni.layer.Dense(64, 64, rand_scaled, minni.activator.Rectifier()))
model.add(minni.layer.Dense(64, 1, rand_scaled, minni.activator.Linear()))

model.train(X, y, epochs=10000)
```

### Classification (Spiral Data)

```python
model = minni.model.Model(
    loss=minni.loss.CrossEntropy(),
    optimizer=minni.optimizer.Adam(eta=0.05, beta=5e-5),
    accuracy=minni.accuracy.Categorical())

model.add(minni.layer.Dense(2, 512, 
    activator=minni.activator.Rectifier(),
    regularizer=minni.regularizer.Ridge(5e-4)))
model.add(minni.layer.Dropout(0.1))
model.add(minni.layer.Dense(512, 3, 
    activator=minni.activator.Softmax()))

model.train(X, y, epochs=10000)
model.evaluate(Xt, yt)
```

### MNIST Fashion Dataset Classification

#### Training

```python
model = minni.model.Model(
    loss=minni.loss.CrossEntropy(),
    optimizer=minni.optimizer.Adam(beta=1e-3),
    accuracy=minni.accuracy.Categorical())
model.add(minni.layer.Dense(X.shape[1], 128, 
    activator=minni.activator.Rectifier()))
model.add(minni.layer.Dense(128, 128, 
    activator=minni.activator.Rectifier()))
model.add(minni.layer.Dense(128, 10, 
    activator=minni.activator.Softmax()))

model.train(X, y, epochs=10, batch_size=128)
model.evaluate(Xt, yt)
model.save('model.minni')
```

#### Testing

```python
model = minni.model.Model.load('model.minni')
model.evaluate(Xt, yt)

# Predict on the test dataset
failures = 0
for i, j in enumerate(model.predict(Xt)):
    if j != yt[i]: failures += 1
print(f'Failures: {failures}')
```

#### Prediction

```python
image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'image.png')
image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Grayscale
image_data = cv2.resize(image_data, (28,28)) # Resize 
image_data = 255 - image_data # Color inversion
image_data = (image_data.reshape(1,-1).astype(numpy.float32) - 127.5) / 127.5

if show:
    matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='gray')
    matplotlib.pyplot.show() # Show the image

model = minni.model.Model.load('mnist_fashion.minni')
yhat = model.predict(image_data)

print(f'Image is predicted as a {MNIST_FASHION_LABELS[yhat[0]]}')
```

## License

This software is licensed under the [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
