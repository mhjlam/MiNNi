import os
import sys
import numpy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import minni
import minni.layer
import minni.loss
import minni.model
import minni.activator
import minni.optimizer
import minni.visualizer
import minni.initializer

minni.init()


def generate_sine_dataset(N=1000):
    X = numpy.arange(N).reshape(-1, 1) / N
    y = numpy.sin(2 * numpy.pi * X).reshape(-1, 1)
    return X, y


if __name__ == '__main__':
    print('Regression (sine)')
    
    X, y = generate_sine_dataset()

    model = minni.model.Model(loss=minni.loss.MeanSquaredError(),
                              optimizer=minni.optimizer.Adam(eta=0.005, beta=0.001),
                              metric=minni.Metric.REGRESSION)
    
    rand_scaled = minni.initializer.Random(scaler=0.1)
    model.add(minni.layer.Dense(1, 64, rand_scaled, minni.activator.Rectifier()))
    model.add(minni.layer.Dense(64, 64, rand_scaled, minni.activator.Rectifier()))
    model.add(minni.layer.Dense(64, 1, rand_scaled, minni.activator.Linear()))

    visualizer = minni.visualizer.Plot(model, save_path="regression.mp4", 
                                       interval=25, fps=30, bitrate=3200)
    visualizer.record(X, y, epochs=1000)
