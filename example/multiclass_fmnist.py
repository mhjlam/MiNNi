import os
import gzip
import struct

import cv2
import numpy
import urllib.request
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import minni
import minni.activator
import minni.layer
import minni.loss
import minni.model
import minni.optimizer

import run_example as example


MODEL_PATH = os.path.join(example.OUTPUT_DIR, "multiclass_fmnist.minni")


def read_dataset(set="train"):  # or t10k
    X = []
    y = []

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
    print("Preprocessing")

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


def train(X, y, plot=False):
    print("\nTraining")

    # Initialize the model
    model = minni.model.Model(loss=minni.loss.CrossEntropy(), optimizer=minni.optimizer.Adam(eta=0.001))
    model.add(minni.layer.Dense(X.shape[1], 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(128, 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(128, 10, activator=minni.activator.Softmax()))

    num_epochs = 10
    batch_size = 128
    num_batches = len(X) // batch_size

    # Metrics storage
    batch_losses = []
    batch_accuracies = []
    batch_indices = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        total = 0
        correct = 0

        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            # Ensure y_batch is a 1D array of integers
            y_batch = y_batch.astype(numpy.int64)

            # Forward pass and compute loss
            forward_output = model.forward(X_batch)

            # If forward_output is a tuple, extract the predictions
            if isinstance(forward_output, tuple):
                y_pred = forward_output[0]  # Assuming the first element is the prediction
            else:
                y_pred = forward_output

            # Ensure y_pred is a 2D array with shape (batch_size, num_classes)
            if y_pred.ndim != 2 or y_pred.shape[0] != X_batch.shape[0]:
                raise ValueError(f"Unexpected shape for y_pred: {y_pred.shape}")

            batch_loss, _ = model.loss(y_pred, y_batch, model.layers)

            # Backward pass and update weights
            model.backward(y_pred, y_batch)
            model.optimizer.optimize(model.layers)

            # Calculate accuracy
            batch_accuracy = numpy.mean(numpy.argmax(y_pred, axis=1) == y_batch)

            # Update metrics
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            
            correct += numpy.sum(numpy.argmax(y_pred, axis=1) == y_batch)
            total += y_batch.size

            # Store batch metrics
            if plot:
                batch_losses.append(batch_loss)
                batch_accuracies.append(correct / total * 100)
                batch_indices.append(epoch + batch / num_batches)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}, Accuracy: {correct / total * 100:.2f}%")

    # Plotting metrics
    if plot:
        plt.figure(figsize=(12, 8))

        # Plot loss per batch
        plt.subplot(2, 1, 1)
        plt.plot(batch_indices, batch_losses, label="Loss", color="blue", linewidth=0.5)
        plt.xticks(ticks=range(num_epochs))
        plt.yticks(ticks=numpy.linspace(0.0, 2.0, num=9))
        plt.ylim(0.0, 2.0)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Plot accuracy per batch
        plt.subplot(2, 1, 2)
        plt.plot(batch_indices, batch_accuracies, label="Accuracy", color="green", linewidth=0.5)
        plt.xticks(ticks=range(num_epochs))
        plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])
        plt.ylim(0, 100)
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")

        plt.tight_layout()
        plt.savefig(os.path.join(example.OUTPUT_DIR, "multiclass_fmnist-loss.png"))
        plt.show()

    # Save the model
    model.save(MODEL_PATH)


def evaluate(Xt, yt, plot=False):
    print("\nEvaluating")

    model = minni.model.Model.load(MODEL_PATH)
    model.evaluate(Xt, yt)

    # Predict on the test dataset
    y_hat = model.predict(Xt)
    failures = sum(y_hat != yt)
    print(f"Failures: {failures}/{Xt.shape[0]} = {failures / Xt.shape[0] * 100:.2f}%")

    if plot:
        # Save confusion matrix to a file
        cm = metrics.confusion_matrix(yt, y_hat)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=example.FMNIST_LABELS)
        disp.plot(cmap="viridis", xticks_rotation=45)
        plt.title("Confusion Matrix")

        # Adjust layout to prevent cropped labels
        plt.tight_layout()
        plt.savefig(os.path.join(example.OUTPUT_DIR, "multiclass_fmnist-error.png"))
        plt.show()


def predict(Xt, yt, images):
    print("\nPredicting")

    model = minni.model.Model.load(MODEL_PATH)
    model.evaluate(Xt, yt)

    for image in images:
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), image)
        if not os.path.exists(image_path):
            continue
        
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)       # Grayscale
        image_data = cv2.resize(image_data, (28, 28))                   # Resize
        image_data = 255 - image_data                                   # Color inversion
        image_data = image_data.reshape(1, -1).astype(numpy.float32)    # Flatten
        image_data = (image_data - 127.5) / 127.5                       # Normalize

        yhat = model.predict(image_data)
        print(f"{image} is predicted as a {example.FMNIST_LABELS[yhat[0]]}")


def main():
    X, y, Xt, yt = preprocess_dataset()

    train(X, y, plot=True)
    evaluate(Xt, yt, plot=True)
    predict(Xt, yt, example.PREDICT_IMAGES)
