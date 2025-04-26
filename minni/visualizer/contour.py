import numpy
import matplotlib.pyplot as plt

from .visualizer import Visualizer


class Contour(Visualizer):
    def __init__(self, model, save_path="contour.mp4", interval=10, fps=30, bitrate=3200):
        super().__init__(model, save_path, interval, fps, bitrate)

    def setup(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor('white')
        self.ax.set_title("Classification")
        self.ax.set_xlim(self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5)
        self.ax.set_ylim(self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5)
        self.ax.axis('off')  # Removes all ticks, labels, and grid

        # Plot the true data points
        self.edge_colors = "white"
        if len(numpy.unique(self.y)) % 2 == 1:  # Check if the number of classes is odd
            middle_class = sorted(numpy.unique(self.y))[len(numpy.unique(self.y)) // 2]
            self.edge_colors = ["black" if label == middle_class else "white" for label in self.y.flatten()]
            
        self.scatter = self.ax.scatter(self.X[:, 0], self.X[:, 1], 
            c=self.y.flatten(), cmap="RdBu", edgecolor=self.edge_colors, s=50)

    def frame(self, _):
        self.model.train(self.X, self.y, epochs=10)
        self.ax.clear()
        self.ax.axis('off')
        
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        
        xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100),
                                numpy.linspace(y_min, y_max, 100))
        
        grid = numpy.c_[xx.ravel(), yy.ravel()]
        probs = self.model.predict(grid).reshape(xx.shape)
        
        self.ax.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.6)
        self.ax.scatter(self.X[:, 0], self.X[:, 1], 
            c=self.y.flatten(), cmap="RdBu", edgecolor=self.edge_colors, s=50)
        
        return self.scatter,
