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

import run_example as example


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
    return X, y.reshape(-1, 1)


def main():
    print("\nLogistic Regression (spiral data)")

    X, y = generate_spiral_dataset(N=100, C=2)

    model = minni.model.Model(loss=minni.loss.BinaryCrossEntropy(), 
                              optimizer=minni.optimizer.Adam(), metric=minni.Metric.BINARY)

    model.add(minni.layer.Dense(2, 64, activator=minni.activator.Rectifier(), 
                                regularizer=minni.regularizer.Ridge(0.0005)))
    
    model.add(minni.layer.Dense(64, 1, activator=minni.activator.Sigmoid()))

    # Train the model with animated plot
    visualizer = minni.visualizer.Contour(model, save_path=os.path.join(example.OUTPUT_DIR, "binclass_spiral.mp4"), 
                                          interval=25, fps=30, bitrate=3200)
    
    visualizer.record(X, y, epochs=1000)
