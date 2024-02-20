import os
import cv2
import numpy

import snn.model
import snn.assets

example_dir = os.path.join(os.path.dirname(__file__))
asset_dir = os.path.abspath(os.path.join(example_dir, '..\\assets'))

# Load previously saved model
model = snn.model.Model.load(os.path.join(example_dir, 'fashion_mnist.snnm'))

def predict_image(image_filename):
    # Read as grayscale
    image_data = cv2.imread(os.path.join(asset_dir, image_filename), cv2.IMREAD_GRAYSCALE)
    
    # Resize image to 28 by 28 pixels
    image_data = cv2.resize(image_data, (28, 28))
    
    # Invert colors
    image_data = 255 - image_data
    
    # Reshape and scale to [-1,1]
    image_data = (image_data.reshape(1,-1).astype(numpy.float32) - 127.5) / 127.5

    # Predict confidence of image
    confidences = model.predict(image_data)

    # Get prediction class from confidence levels
    predictions = model.output_activation.predictions(confidences)

    # Prediction label
    print(image_filename + '\t' + snn.assets.FASHION_MNIST_LABELS[predictions[0]])

if __name__ == '__main__':
    predict_image('tshirt.png')
    predict_image('pants.png')
