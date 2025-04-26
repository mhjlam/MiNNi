import numpy
import matplotlib.pyplot as plt

from .visualizer import Visualizer


class Plot(Visualizer):
    def __init__(self, model, save_path="plot.mp4", interval=10, fps=30, bitrate=3200):
        super().__init__(model, save_path, interval, fps, bitrate)


    def setup(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.patch.set_facecolor('white')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axis('off') # Removes all ticks, labels, and grid

        # Plot the true sine curve
        self.ax.plot(self.X, self.y, color='black', linewidth=2, label='Sine')
        
        # Initialize the predicted curve line
        self.line, = self.ax.plot(self.X, numpy.zeros_like(self.y), color="red", linewidth=2, alpha=0.75)


    def frame(self, _):
        self.model.train(self.X, self.y, epochs=1)
        y_pred = self.model.predict(self.X)
        self.line.set_ydata(y_pred)
        return self.line,
