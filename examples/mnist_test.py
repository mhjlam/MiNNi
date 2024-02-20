import os

import snn
import snn.loss
import snn.layer
import snn.model
import snn.assets
import snn.accuracy
import snn.optimizer
import snn.activation

asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\assets'))
fashion_mnist_images_dir = os.path.join(asset_dir, 'fashion_mnist_images')

# Import dataset (if necessary)
snn.assets.import_dataset_mnist(fashion_mnist_images_dir)

# Load test dataset
X_test, y_test = snn.assets.load_dataset_mnist('test', fashion_mnist_images_dir)

# Preprocess test dataset
X_test = snn.assets.preprocess_test_dataset_mnist(X_test)

# Load previously saved model
model = snn.model.Model.load(os.path.join(os.path.dirname(__file__), 'fashion_mnist.snnm'))

# Evaluate the model
model.evaluate(X_test, y_test)

# Predict on the first 5 samples from validation dataset
confidences = model.predict(X_test[:5])
predictions = model.output_activation.predictions(confidences)

for prediction in predictions:
    print(prediction, snn.assets.FASHION_MNIST_LABELS[prediction])
