# Mini Neural Network

## Example Usage

### Regression

```python
model = mnn.model.Model(loss=mnn.loss.MeanSquaredError(),
                        optimizer=mnn.optimizer.Adam(eta=0.005, beta=1e-3),
                        accuracy=mnn.accuracy.Regression())

rand_scaled = mnn.initializer.Random(scaler=0.1)
model.add(mnn.layer.Dense(1, 64, rand_scaled, mnn.activator.Rectifier()))
model.add(mnn.layer.Dense(64, 64, rand_scaled, mnn.activator.Rectifier()))
model.add(mnn.layer.Dense(64, 1, rand_scaled, mnn.activator.Linear()))

model.train(X, y, epochs=10000)
```

### Classification (Spiral Data)

```python
model = mnn.model.Model(loss=mnn.loss.SoftmaxLoss(),
                        optimizer=mnn.optimizer.Adam(eta=0.05, beta=5e-5),
                        accuracy=mnn.accuracy.Categorical())
model.add(mnn.layer.Dense(2, 512, activator=mnn.activator.Rectifier(),
                          regularizer=mnn.regularizer.Ridge(5e-4)))
model.add(mnn.layer.Dropout(0.1))
model.add(mnn.layer.Dense(512, 3, activator=mnn.activator.Softmax()))

model.train(X, y, epochs=10000)
model.evaluate(Xt, yt)
```

### MNIST Fashion Dataset Classification

#### Training

```python
model = mnn.model.Model(loss=mnn.loss.SoftmaxLoss(),
                        optimizer=mnn.optimizer.Adam(beta=1e-3),
                        accuracy=mnn.accuracy.Categorical())
model.add(mnn.layer.Dense(X.shape[1], 128, activator=mnn.activator.Rectifier()))
model.add(mnn.layer.Dense(128, 128, activator=mnn.activator.Rectifier()))
model.add(mnn.layer.Dense(128, 10, activator=mnn.activator.Softmax()))

model.train(X, y, epochs=10, batch_size=128)
model.evaluate(Xt, yt)
model.save('model.mnnm')
```

#### Testing

```python
model = mnn.model.Model.load('model.mnnm')
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

model = mnn.model.Model.load('mnist_fashion.mnnm')
yhat = model.predict(image_data)

print(f'Image is predicted as a {MNIST_FASHION_LABELS[yhat[0]]}')
```
