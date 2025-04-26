import os

import cv2
import numpy
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import minni
import minni.layer
import minni.loss
import minni.model
import minni.activator
import minni.optimizer


DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(DIR, os.pardir))
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "_output")

SHAPES = {
    "circle",
    "square",
    "triangle",
}


def generate_shape_image(shape, img_size=28):
    img = numpy.zeros((img_size, img_size), dtype=numpy.uint8)
    center = (numpy.random.randint(8, 20), numpy.random.randint(8, 20))
    size = numpy.random.randint(6, 10)

    if shape == "circle":
        cv2.circle(img, center, size, 255, -1)
    elif shape == "square":
        top_left = (center[0] - size, center[1] - size)
        bottom_right = (center[0] + size, center[1] + size)
        cv2.rectangle(img, top_left, bottom_right, 255, -1)
    elif shape == "triangle":
        pt1 = (center[0], center[1] - size)
        pt2 = (center[0] - size, center[1] + size)
        pt3 = (center[0] + size, center[1] + size)
        pts = numpy.array([pt1, pt2, pt3], numpy.int32).reshape((-1, 1, 2))
        cv2.drawContours(img, [pts], 0, 255, -1)

    # Normalize and add slight noise
    img = img.astype(numpy.float32) / 255.0
    img += numpy.random.normal(0, 0.05, img.shape)
    return numpy.clip(img, 0.0, 1.0)


def generate_shape_dataset(num_samples_per_class=1000, img_size=28):
    X = []
    y = []

    for label, shape in enumerate(SHAPES):
        for _ in range(num_samples_per_class):
            X.append(generate_shape_image(shape, img_size))
            y.append(label)

    X = numpy.array(X, dtype=numpy.float32).reshape(len(X), 1, img_size, img_size)
    y = numpy.array(y, dtype=numpy.int32)

    # Shuffle dataset
    indices = numpy.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":
    X, y = generate_shape_dataset(num_samples_per_class=1000)
    Xt, yt = generate_shape_dataset(num_samples_per_class=100)

    model = minni.model.Model(loss=minni.loss.CrossEntropy(), optimizer=minni.optimizer.Adam(eta=0.001))
    model.add(minni.layer.Conv(1, 32, kernel_size=3, stride=1, padding=1, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Pooling(pool_size=2, stride=2))
    model.add(minni.layer.Dropout(0.25))
    model.add(minni.layer.Conv(32, 64, kernel_size=3, stride=1, padding=1, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Pooling(pool_size=2, stride=2))
    model.add(minni.layer.Dropout(0.25))
    model.add(minni.layer.Flatten((64, 7, 7), 64 * 7 * 7))
    model.add(minni.layer.Dense(64 * 7 * 7, 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dropout(0.5))
    model.add(minni.layer.Dense(128, 10, activator=minni.activator.Softmax()))

    # Train the model
    model.train(X, y, epochs=10, batch_size=32)

    # Evaluate the model
    model.evaluate(Xt, yt)

    # Predict on evaluation dataset
    y_pred = model.predict(Xt)

    # Generate confusion matrix
    cm = metrics.confusion_matrix(yt, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SHAPES)

    # Plot and save confusion matrix
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn", "shapes-error.png"))
    plt.close()
