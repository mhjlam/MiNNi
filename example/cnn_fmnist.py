import os
import gzip
import struct

import cv2
import numpy
import urllib.request
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import minni
import minni.layer
import minni.loss
import minni.model
import minni.activator
import minni.optimizer

import run_example as example


def read_dataset(set="train"):  # or t10k
    image_set_file = f"{set}-images-idx3-ubyte.gz"
    label_set_file = f"{set}-labels-idx1-ubyte.gz"

    images_gz = os.path.join(example.FASHION_MNIST_RAW_DIR, image_set_file)
    labels_gz = os.path.join(example.FASHION_MNIST_RAW_DIR, label_set_file)

    # Download the dataset zips if it does not exist
    if not os.path.exists(images_gz):
        urllib.request.urlretrieve(f"{example.FMNIST_URL}/{set}-images-idx3-ubyte.gz", images_gz)

    if not os.path.exists(labels_gz):
        urllib.request.urlretrieve(f"{example.FMNIST_URL}/{set}-labels-idx1-ubyte.gz", labels_gz)

    with gzip.open(images_gz, "rb") as f:
        # Read the first 16 bytes (magic number, count, rows, cols)
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))

        # Read the image data (one byte per pixel)
        X = numpy.frombuffer(f.read(), dtype=numpy.uint8).reshape(num, rows, cols)

    with gzip.open(labels_gz, "rb") as f:
        # Read the first 8 bytes (magic number and number of labels)
        _, num = struct.unpack(">II", f.read(8))

        # Read the rest as label data (one byte per label)
        y = numpy.frombuffer(f.read(), dtype=numpy.uint8)

    return X, y


def preprocess_dataset():
    X, y = read_dataset("train")
    Xt, yt = read_dataset("t10k")

    # Shuffle training dataset
    keys = numpy.array(range(X.shape[0]))
    numpy.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # Flatten matrices and scale to [-1,1] range
    X = (X.reshape(X.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5
    Xt = (Xt.reshape(Xt.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5

    return X, y, Xt, yt


def train_predict(X, y, Xt, yt):
    # Reshape input data for Conv layer (batch_size, height, width, channels)
    X = X.reshape(-1, 28, 28, 1)  # Add channel dimension
    Xt = Xt.reshape(-1, 28, 28, 1)

    model = minni.model.Model(loss=minni.loss.CrossEntropy(), optimizer=minni.optimizer.Adam(eta=0.001))

    # Convolutional layer 1
    model.add(minni.layer.Conv(1, 32, kernel_size=3, stride=1, padding=1, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Pooling(pool_size=2, stride=2))
    model.add(minni.layer.Dropout(0.25))

    # Convolutional layer 2
    model.add(minni.layer.Conv(32, 64, kernel_size=3, stride=1, padding=1, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Pooling(pool_size=2, stride=2))
    model.add(minni.layer.Dropout(0.25))

    # Flatten and add dense layers
    model.add(minni.layer.Flatten((64, 7, 7), 64 * 7 * 7))

    model.add(minni.layer.Dense(64 * 7 * 7, 64, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(64, 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dropout(0.5))
    model.add(minni.layer.Dense(128, 10, activator=minni.activator.Softmax()))

    # Train the model
    batch_size = len(X) // 1000
    print("Batch size: ", batch_size)
    model.train(X, y, epochs=1, batch_size=batch_size)

    # Evaluate the model
    model.evaluate(Xt, yt)

    # Predict on evaluation dataset
    y_pred = model.predict(Xt)

    # Generate confusion matrix
    cm = metrics.confusion_matrix(yt, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(example.FMNIST_LABELS.values()))

    # Plot and save confusion matrix
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(example.OUTPUT_DIR, "cnn_fmnist-error.png"))
    plt.close()

    for image_path in example.PREDICT_IMAGES:
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)       # Grayscale
        image_data = cv2.resize(image_data, (28, 28))                   # Resize
        image_data = 255 - image_data                                   # Color inversion
        image_data = image_data.reshape(-1, 28, 28, 1)                  # Add channel dimension
        image_data = (image_data.astype(numpy.float32) - 127.5) / 127.5 # Normalize

        yhat = model.predict(image_data)
        print(f"{image_path} is predicted as a {example.FMNIST_LABELS[yhat[0]]}")


def main():
    X, y, Xt, yt = preprocess_dataset()
    train_predict(X, y, Xt, yt)
