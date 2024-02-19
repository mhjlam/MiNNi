import os
import cv2
import numpy
import random
import urllib
import zipfile
import urllib.request
import matplotlib.pyplot

URL = 'http://nnfs.io/datasets/fashion_mnist_images.zip'

def import_dataset_mnist(target_dir):
    if os.path.exists(target_dir):
        return
    
    print('Importing dataset')
        
    datadir = os.path.dirname(__file__)
    file = os.path.join(datadir, 'fashion_mnist_images.zip')
    dir = os.path.join(datadir, 'fashion_mnist_images')

    if not os.path.exists(file):
        print(f'Downloading {URL} and saving as {file}...')
        urllib.request.urlretrieve(URL, file)

    print('Extracting zipfile...')
    with zipfile.ZipFile(file) as zip_images:
        zip_images.extractall(dir)

def load_dataset_mnist(dataset, path):
    print(f'Loading dataset ({dataset})')
    X = []
    y = []
    for label in os.listdir(os.path.join(path, dataset)):
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return numpy.array(X), numpy.array(y).astype('uint8')

def preprocess_dataset_mnist(X, y, X_test, y_test):
    print('Preprocessing')
    
    # Shuffle training dataset
    keys = numpy.array(range(X.shape[0]))
    numpy.random.shuffle(keys)
    X = X[keys]
    y = y[keys]
    
    # Flatten matrix and scale to [-1,1] range
    X = (X.reshape(X.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5
    
    return X, y, X_test, y_test

def show_random_training_image(dir):
    random_class = random.randint(0,9)
    random_image = random.randint(0,999)

    filepath = os.path.join(dir, 'train', f'{random_class}', f'{random_image:04d}.png')
    print(filepath)
    matplotlib.pyplot.imshow(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), cmap='gray') 
    matplotlib.pyplot.show()
    
if __name__ == '__main__':
    import_fashion_mnist_images()
