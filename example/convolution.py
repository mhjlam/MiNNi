import os
import sys
import gzip
import numpy
import struct
import urllib.request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import minni
import minni.activator
import minni.initializer
import minni.layer
import minni.loss
import minni.model
import minni.optimizer

minni.init()

MDL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'conv.mdl')


def read_mnist_fashion_dataset(set='train'): # or t10k
    X = []
    y = []
    
    image_set_file = f'{set}-images-idx3-ubyte.gz'
    label_set_file = f'{set}-labels-idx1-ubyte.gz'

    mnist_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fashion_mnist')
    images_gz = os.path.join(mnist_folder, image_set_file)
    labels_gz = os.path.join(mnist_folder, label_set_file)
    
    if not os.path.exists(mnist_folder):
        os.makedirs(mnist_folder, exist_ok=True)
    
    # Download the dataset zips if it does not exist
    if not os.path.exists(image_set_file):
        if not os.path.exists(images_gz):
            urllib.request.urlretrieve(
                f'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{set}-images-idx3-ubyte.gz', 
                images_gz)
    
    if not os.path.exists(label_set_file):
        if not os.path.exists(labels_gz):
            urllib.request.urlretrieve(
                f'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{set}-labels-idx1-ubyte.gz', 
                labels_gz)

    with gzip.open(images_gz, 'rb') as f:
        # Read the first 16 bytes (magic number, count, rows, cols)
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        
        # Read the image data (one byte per pixel)
        X = numpy.frombuffer(f.read(), dtype=numpy.uint8).reshape(num, rows, cols)
    
    with gzip.open(labels_gz, 'rb') as f:
        # Read the first 8 bytes (magic number and number of labels)
        _, num = struct.unpack('>II', f.read(8))
        
        # Read the rest as label data (one byte per label)
        y = numpy.frombuffer(f.read(), dtype=numpy.uint8)

    return X, y


def preprocess_mnist_fashion_dataset():
    X, y = read_mnist_fashion_dataset('train')
    Xt, yt = read_mnist_fashion_dataset('t10k')
    
    # Shuffle training dataset
    keys = numpy.array(range(X.shape[0]))
    numpy.random.shuffle(keys)
    X = X[keys]
    y = y[keys]
    
    # Flatten matrices and scale to [-1,1] range
    X = (X.reshape(X.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5
    Xt = (Xt.reshape(Xt.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5

    return X, y, Xt, yt


def convolution_train(X, y, Xt, yt):
    # Reshape input data for Conv2D (batch_size, height, width, channels)
    X = X.reshape(-1, 28, 28, 1)  # Add channel dimension
    Xt = Xt.reshape(-1, 28, 28, 1)
    
    model = minni.model.Model(loss=minni.loss.SoftmaxLoss(),
                              optimizer=minni.optimizer.Adam(beta=1e-3))
    
    # Add convolutional layers
    model.add(minni.layer.Conv(1, 16, kernel_size=3, stride=1, padding=1, initializer=minni.initializer.Random()))
    model.add(minni.activator.Rectifier())
    model.add(minni.layer.MaxPooling(pool_size=2, stride=2))
    
    model.add(minni.layer.Conv(16, 32, kernel_size=3, stride=1, padding=1, initializer=minni.initializer.Random()))
    model.add(minni.activator.Rectifier())
    model.add(minni.layer.MaxPooling(pool_size=2, stride=2))
    
    # Flatten and add dense layers
    model.add(minni.layer.Flatten())
    model.add(minni.layer.Dense(32 * 7 * 7, 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(128, 10, activator=minni.activator.Softmax()))
    
    # Train the model
    model.train(X, y, epochs=10, batch_size=128)
    
    # Evaluate the model
    model.evaluate(Xt, yt)
    
    # Save the trained model
    model.save(MDL_PATH)


if __name__ == '__main__':
    print('Convolutional Neural Network (MNIST Fashion)')
    X, y, Xt, yt = preprocess_mnist_fashion_dataset()
    
    model = minni.model.Model(loss=minni.loss.CategoricalCrossEntropy(), optimizer=minni.optimizer.Adam())
    model.add(minni.layer.Conv(3, 16, kernel_size=3, stride=1, padding=1, initializer=minni.initializer.Random()))
    model.add(minni.activator.Rectifier())
    model.add(minni.layer.MaxPooling(pool_size=2, stride=2))
    model.add(minni.layer.Flatten())
    model.add(minni.layer.Dense(16*16*16, 10, activator=minni.Softmax()))
