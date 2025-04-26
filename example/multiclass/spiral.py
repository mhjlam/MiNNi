import os
import numpy

import minni
import minni.loss
import minni.layer
import minni.model
import minni.activator
import minni.optimizer
import minni.visualizer
import minni.regularizer


DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(DIR, os.pardir))
INPUT_DIR = os.path.join(EXAMPLE_DIR, "_input")
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "_output")


def generate_spiral_dataset(N, C):
    numpy.random.seed(420)
    X = numpy.zeros((N * C, 2))
    y = numpy.zeros(N * C, dtype="uint8")
    for c in range(C):
        i = range(N * c, N * (c + 1))
        r = numpy.linspace(0.0, 1, N)
        t = numpy.linspace(c * 4, (c + 1) * 4, N) + numpy.random.randn(N) * 0.2
        X[i] = numpy.c_[r * numpy.sin(t * 2.5), r * numpy.cos(t * 2.5)]
        y[i] = c
    return X, y


if __name__ == "__main__":
    print("\nClassification (spiral data)")

    classes = 3
    samples = 100

    X, y = generate_spiral_dataset(N=samples, C=classes)
    Xt, yt = generate_spiral_dataset(N=samples, C=classes)

    model = minni.model.Model(loss=minni.loss.CrossEntropy(), optimizer=minni.optimizer.Adam(eta=0.05, beta=0.00005))
    
    model.add(minni.layer.Dense(2, 512, activator=minni.activator.Rectifier(), 
                                regularizer=minni.regularizer.Ridge(0.0005)))
    model.add(minni.layer.Dropout(0.1))
    model.add(minni.layer.Dense(512, classes, activator=minni.activator.Softmax()))

    visualizer = minni.visualizer.Contour(model,
        save_path=os.path.join(OUTPUT_DIR, "multiclass", f"spiral-{classes}-fit.mp4"),
        interval=25, fps=30, bitrate=3200)
    
    visualizer.record(X, y, epochs=1000)
