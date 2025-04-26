import os
import numpy

import minni
import minni.layer
import minni.loss
import minni.model
import minni.activator
import minni.optimizer
import minni.visualizer


DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(DIR, os.pardir))
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "_output")


def generate_sine_dataset(N=1000):
    X = numpy.arange(N).reshape(-1, 1) / N
    y = numpy.sin(2 * numpy.pi * X).reshape(-1, 1)
    return X, y


if __name__ == "__main__":
    print("Regression (sine)")

    X, y = generate_sine_dataset(N=1000)

    model = minni.model.Model(loss=minni.loss.MeanSquaredError(), 
                              optimizer=minni.optimizer.Adam(), 
                              metric=minni.Metric.REGRESSION)

    model.add(minni.layer.Dense(1, 256, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(256, 256, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(256, 1, activator=minni.activator.Linear()))

    visualizer = minni.visualizer.Plot(model, save_path=os.path.join(OUTPUT_DIR, "regression", "sone.mp4"), interval=25, fps=30, bitrate=3200)
    visualizer.record(X, y, epochs=1000)
