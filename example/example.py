import os
import cv2
import gzip
import numpy
import urllib.request
import matplotlib.pyplot

import mnn
import mnn.accuracy
import mnn.activator
import mnn.initializer
import mnn.layer
import mnn.loss
import mnn.model
import mnn.optimizer
import mnn.regularizer

mnn.init()

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

MNNM_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist_fashion.mnnm')

def generate_sine_dataset(N=1000):
    X = numpy.arange(N).reshape(-1, 1) / N
    y = numpy.sin(2 * numpy.pi * X).reshape(-1, 1)
    return X, y

def generate_spiral_dataset(N, C):
    numpy.random.seed(420)
    X = numpy.zeros((N*C, 2))
    y = numpy.zeros(N*C, dtype='uint8')
    for c in range(C):
        i = range(N*c, N*(c+1))
        r = numpy.linspace(0.0, 1, N)
        t = numpy.linspace(c*4, (c+1)*4, N) + numpy.random.randn(N)*0.2
        X[i] = numpy.c_[r*numpy.sin(t*2.5), r*numpy.cos(t*2.5)]
        y[i] = c
    return X, y

# def load_mnist_fashion_dataset(set='train'): #t10k
#     dir_path = os.path.dirname(os.path.realpath(__file__))
    
#     X_path = os.path.join(dir_path, f'{set}-images-idx3-ubyte.gz')
#     y_path = os.path.join(dir_path, f'{set}-labels-idx1-ubyte.gz')
    
#     if not os.path.exists(X_path):
#         urllib.request.urlretrieve(f'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{set}-images-idx3-ubyte.gz', X_path)
#     if not os.path.exists(y_path):
#         urllib.request.urlretrieve(f'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{set}-labels-idx1-ubyte.gz', y_path)
    
#     with gzip.open(y_path, 'rb') as f:
#         y = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=8)
    
#     with gzip.open(X_path, 'rb') as f:
#         X = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=16).reshape(-1, 28*28)

#     return X, y

def load_mnist_fashion_dataset(set='train'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    set_path = os.path.join(dir_path, 'mnist_fashion_dataset', set)
    X = []
    y = []
    for label in os.listdir(set_path):
        for file in os.listdir(os.path.join(set_path, label)):
            image = cv2.imread(os.path.join(set_path, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return numpy.array(X), numpy.array(y).astype('uint8')


def preprocess_mnist_fashion_dataset():
    X, y = load_mnist_fashion_dataset('train')
    Xt, yt = load_mnist_fashion_dataset('test')
    
    # Shuffle training dataset
    keys = numpy.array(range(X.shape[0]))
    numpy.random.shuffle(keys)
    X = X[keys]
    y = y[keys]
    
    # Flatten matrices and scale to [-1,1] range
    X = (X.reshape(X.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5
    Xt = (Xt.reshape(Xt.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5

    return X, y, Xt, yt


def regression():
    X, y = generate_sine_dataset()
    
    model = mnn.model.Model(loss=mnn.loss.MeanSquaredError(),
                             optimizer=mnn.optimizer.Adam(eta=0.005, beta=1e-3),
                             accuracy=mnn.accuracy.Regression())
    
    rand_scaled = mnn.initializer.Random(scaler=0.1)
    model.add(mnn.layer.Dense(1, 64, rand_scaled, mnn.activator.Rectifier()))
    model.add(mnn.layer.Dense(64, 64, rand_scaled, mnn.activator.Rectifier()))
    model.add(mnn.layer.Dense(64, 1, rand_scaled, mnn.activator.Linear()))
    
    model.train(X, y, epochs=10000)

def logistic_regression():
    X, y = generate_spiral_dataset(N=100, C=2)
    Xt, yt = generate_spiral_dataset(N=100, C=2)
    
    y = y.reshape(-1, 1)
    yt = yt.reshape(-1, 1)
    
    model = mnn.model.Model(loss=mnn.loss.BinaryCrossEntropy(),
                             optimizer=mnn.optimizer.Adam(beta=5e-7),
                             accuracy=mnn.accuracy.BinaryCategorical())
    
    model.add(mnn.layer.Dense(2, 64, activator=mnn.activator.Rectifier(), 
                               regularizer=mnn.regularizer.Ridge(5e-4)))
    model.add(mnn.layer.Dense(64, 1, activator=mnn.activator.Sigmoid()))
    
    model.train(X, y, epochs=10000)
    model.evaluate(Xt, yt)

def classification():
    X, y = generate_spiral_dataset(N=1000, C=3)
    Xt, yt = generate_spiral_dataset(N=100, C=3)
    
    model = mnn.model.Model(loss=mnn.loss.SoftmaxLoss(),
                            optimizer=mnn.optimizer.Adam(eta=0.05, beta=5e-5),
                            accuracy=mnn.accuracy.Categorical())
    model.add(mnn.layer.Dense(2, 512, activator=mnn.activator.Rectifier(),
                              regularizer=mnn.regularizer.Ridge(5e-4)))
    model.add(mnn.layer.Dropout(0.1))
    model.add(mnn.layer.Dense(512, 3, activator=mnn.activator.Softmax()))
    
    model.train(X, y, epochs=10000)
    model.evaluate(Xt, yt)

def mnist_fashion_train(X, y, Xt, yt):
    model = mnn.model.Model(loss=mnn.loss.SoftmaxLoss(),
                            optimizer=mnn.optimizer.Adam(beta=1e-3),
                            accuracy=mnn.accuracy.Categorical())
    model.add(mnn.layer.Dense(X.shape[1], 128, activator=mnn.activator.Rectifier()))
    model.add(mnn.layer.Dense(128, 128, activator=mnn.activator.Rectifier()))
    model.add(mnn.layer.Dense(128, 10, activator=mnn.activator.Softmax()))
    
    model.train(X, y, epochs=10, batch_size=128)
    model.evaluate(Xt, yt)
    model.save(MNNM_PATH)

def mnist_fashion_test(Xt, yt):
    model = mnn.model.Model.load(MNNM_PATH)
    model.evaluate(Xt, yt)
    
    # Predict on the test dataset
    failures = 0
    for i, j in enumerate(model.predict(Xt)):
        if j != yt[i]: failures += 1
    print(f'Failures: {failures}')
    
def mnist_fashion_predict(Xt, yt, image, show=False):
    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), image)
    image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Grayscale
    image_data = cv2.resize(image_data, (28,28)) # Resize 
    image_data = 255 - image_data # Color inversion
    #image_data = image_data.reshape(-1, 28*28).astype(numpy.float32)
    image_data = (image_data.reshape(1,-1).astype(numpy.float32) - 127.5) / 127.5
    
    if show:
        matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='gray')
        matplotlib.pyplot.show() # Show the image
    
    model = mnn.model.Model.load(MNNM_PATH)
    model.evaluate(Xt, yt)
    
    yhat = model.predict(image_data)
    print(f'{image} is predicted as a {MNIST_FASHION_LABELS[yhat[0]]}')


if __name__ == '__main__':
    print('Regression (sine)')
    regression()
    
    print('\nLogistic Regression (spiral data)')
    logistic_regression()
    
    print('\nClassification (spiral data)')
    classification()
    
    X, y, Xt, yt = preprocess_mnist_fashion_dataset()
    
    print('\nClassification (MNIST Fashion train)')
    mnist_fashion_train(X, y, Xt, yt)
     
    print('\nClassification (MNIST Fashion test)')
    mnist_fashion_test(Xt, yt)
    
    print('\nClassification (MNIST Fashion predict)')
    mnist_fashion_predict(Xt, yt, 'tshirt.png')
    mnist_fashion_predict(Xt, yt, 'pants.png')
