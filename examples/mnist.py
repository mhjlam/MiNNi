import os

import snn
import snn.loss
import snn.layer
import snn.model
import snn.assets
import snn.accuracy
import snn.optimizer
import snn.activation

# Import dataset
mnist_fashion_images_dir = \
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\assets\\fashion_mnist_images'))
snn.assets.import_dataset_mnist(mnist_fashion_images_dir)

# Load dataset
X, y = snn.assets.load_dataset_mnist('train', mnist_fashion_images_dir)
X_test, y_test = snn.assets.load_dataset_mnist('test', mnist_fashion_images_dir)

# Preprocess dataset
X, y, X_test, y_test = snn.assets.preprocess_dataset_mnist(X, y, X_test, y_test)

# Define model
model = snn.model.Model()
model.add(snn.layer.Dense(X.shape[1], 128))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(128, 128))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(128, 10))
model.add(snn.activation.Softmax())

model.set(
    loss_func=snn.loss.CategoricalCrossEntropy(),
    optimizer=snn.optimizer.Adam(decay=1e-3),
    accuracy=snn.accuracy.Categorical()
)    
model.finalize()

# Train model
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_freq=100)
