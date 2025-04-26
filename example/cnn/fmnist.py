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


DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(DIR, os.pardir))
INPUT_DIR = os.path.join(EXAMPLE_DIR, "_input")
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "_output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "fmnist_cnn.minni")

FMNIST_LABELS = {
    0: "Shirt (casual)",
    1: "Trousers",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt (formal)",
    7: "Sneaker",
    8: "Bag",
    9: "Boot",
}

PREDICT_IMAGES = [
    os.path.join(INPUT_DIR, "bag.png"),
    os.path.join(INPUT_DIR, "boot.png"),
    os.path.join(INPUT_DIR, "coat.png"),
    os.path.join(INPUT_DIR, "dress.png"),
    os.path.join(INPUT_DIR, "jeans.png"),
    os.path.join(INPUT_DIR, "pullover.png"),
    os.path.join(INPUT_DIR, "shirt.png"),
    os.path.join(INPUT_DIR, "sneaker.png"),
    os.path.join(INPUT_DIR, "tshirt.png"),
]


def read_dataset(set="train"):  # or t10k
    image_set_file = f"{set}-images-idx3-ubyte.gz"
    label_set_file = f"{set}-labels-idx1-ubyte.gz"

    images_gz = os.path.join(INPUT_DIR, image_set_file)
    labels_gz = os.path.join(INPUT_DIR, label_set_file)

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)

    # Download the dataset zips if it does not exist
    if not os.path.exists(images_gz):
        urllib.request.urlretrieve(
            f"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{set}-images-idx3-ubyte.gz", images_gz)

    if not os.path.exists(labels_gz):
        urllib.request.urlretrieve(
            f"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{set}-labels-idx1-ubyte.gz", labels_gz)

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
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(FMNIST_LABELS.values()))

    # Plot and save confusion matrix
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn", "fmnist-error.png"))
    plt.close()

    for image_path in PREDICT_IMAGES:
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)       # Grayscale
        image_data = cv2.resize(image_data, (28, 28))                   # Resize
        image_data = 255 - image_data                                   # Color inversion
        image_data = image_data.reshape(-1, 28, 28, 1)                  # Add channel dimension
        image_data = (image_data.astype(numpy.float32) - 127.5) / 127.5 # Normalize

        yhat = model.predict(image_data)
        print(f"{image_path} is predicted as a {FMNIST_LABELS[yhat[0]]}")

if __name__ == "__main__":
    X, y, Xt, yt = preprocess_dataset()
    train_predict(X, y, Xt, yt)
