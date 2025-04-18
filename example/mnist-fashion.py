import os
import cv2
import sys
import gzip
import numpy
import struct
import urllib.request
import matplotlib.pyplot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import minni
import minni.activator
import minni.layer
import minni.loss
import minni.model
import minni.optimizer

minni.init()

MNIST_FASHION_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

MDL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist_fashion.mdl')


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


def mnist_fashion_train(X, y, Xt, yt):
    model = minni.model.Model(loss=minni.loss.SoftmaxLoss(),
                              optimizer=minni.optimizer.Adam(beta=1e-3))
    model.add(minni.layer.Dense(X.shape[1], 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(128, 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(128, 10, activator=minni.activator.Softmax()))
    
    model.train(X, y, epochs=10, batch_size=128)
    model.evaluate(Xt, yt)
    model.save(MDL_PATH)


def mnist_fashion_test(Xt, yt):
    model = minni.model.Model.load(MDL_PATH)
    model.evaluate(Xt, yt)
    
    # Predict on the test dataset
    failures = 0
    for i, j in enumerate(model.predict(Xt)):
        if j != yt[i]: failures += 1
    print(f'Failures: {failures}')


def mnist_fashion_predict(Xt, yt, images, show=False):
    model = minni.model.Model.load(MDL_PATH)
    model.evaluate(Xt, yt)
    
    for image in images:    
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), image)
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Grayscale
        image_data = cv2.resize(image_data, (28,28)) # Resize 
        image_data = 255 - image_data # Color inversion
        #image_data = image_data.reshape(-1, 28*28).astype(numpy.float32)
        image_data = (image_data.reshape(1,-1).astype(numpy.float32) - 127.5) / 127.5
        
        if show:
            matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='gray')
            matplotlib.pyplot.show() # Show the image
    
        yhat = model.predict(image_data)
        print(f'{image} is predicted as a {MNIST_FASHION_LABELS[yhat[0]]}')


if __name__ == '__main__':
    print('Preprocessing MNIST Fashion data set')
    X, y, Xt, yt = preprocess_mnist_fashion_dataset()
    
    print('\nClassification (MNIST Fashion train)')
    mnist_fashion_train(X, y, Xt, yt)
     
    print('\nClassification (MNIST Fashion test)')
    mnist_fashion_test(Xt, yt)
    
    print('\nClassification (MNIST Fashion predict)')
    mnist_fashion_predict(Xt, yt, ['tshirt.png', 'pants.png'])
