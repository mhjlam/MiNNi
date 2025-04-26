import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .visualizer import Visualizer


class WeightVisualizer(Visualizer):
    def __init__(self, model, save_path=None, interval=100, fps=30, bitrate=1800):
        super().__init__(model, save_path, interval, fps, bitrate)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor('white')
        self.ax.axis('off')  # Completely remove axes
        self.ax.set_xticks([])  # Remove x-axis ticks
        self.ax.set_yticks([])  # Remove y-axis ticks
        self.ax.spines['top'].set_visible(False)  # Remove top spine
        self.ax.spines['right'].set_visible(False)  # Remove right spine
        self.ax.spines['left'].set_visible(False)  # Remove left spine
        self.ax.spines['bottom'].set_visible(False)  # Remove bottom spine
        self.node_positions = []
        self.texts = []
        self.lines = []

    def _initialize_plot(self):
        self.ax.clear()
        self.ax.axis('off')  # Ensure axes are off
        self.ax.set_xticks([])  # Remove x-axis ticks
        self.ax.set_yticks([])  # Remove y-axis ticks
        self.ax.spines['top'].set_visible(False)  # Remove top spine
        self.ax.spines['right'].set_visible(False)  # Remove right spine
        self.ax.spines['left'].set_visible(False)  # Remove left spine
        self.ax.spines['bottom'].set_visible(False)  # Remove bottom spine
        self.node_positions = []
        self.texts = []
        self.lines = []

        num_layers = len(self.model.layers) + 1  # Include input layer
        layer_positions = numpy.linspace(0.1, 0.9, num_layers)  # Horizontal positions of layers

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                num_nodes = self.model.layers[0].W.shape[0]  # Input layer nodes
            else:
                num_nodes = self.model.layers[layer_idx - 1].W.shape[1]  # Hidden/output layer nodes

            if num_nodes > 5:
                nodes_to_display = [0, 1, 2, num_nodes - 3, num_nodes - 2, num_nodes - 1]
            else:
                nodes_to_display = list(range(num_nodes))

            # Assign vertical positions based on node index (top to bottom)
            node_positions = numpy.linspace(0.9, 0.1, num_nodes)  # Top to bottom
            displayed_positions = [node_positions[node_id] for node_id in nodes_to_display]
            self.node_positions.append((layer_positions[layer_idx], displayed_positions))

            # Plot nodes
            for i, node_id in enumerate(nodes_to_display):
                y = displayed_positions[i]
                circle = plt.Circle(
                    (layer_positions[layer_idx], y), 
                    0.03, 
                    color='#4CAF50',  # UI-friendly green color
                    edgecolor='black',  # Black border
                    linewidth=1.5, 
                    alpha=0.9
                )
                self.ax.add_artist(circle)
                self.texts.append(self.ax.text(
                    layer_positions[layer_idx], y, str(node_id),
                    color='white', ha='center', va='center', fontsize=10
                ))

            # Plot connections
            if layer_idx > 0:
                prev_x, prev_y_positions = self.node_positions[layer_idx - 1]
                for i, y in enumerate(displayed_positions):
                    for j, py in enumerate(prev_y_positions):
                        line, = self.ax.plot([prev_x, layer_positions[layer_idx]], [py, y], 
                                             color='gray', alpha=0.5, linewidth=1)
                        self.lines.append(line)

    def _update_weights(self):
        for layer_idx, layer in enumerate(self.model.layers):
            num_nodes = layer.W.shape[1]
            if num_nodes > 5:
                nodes_to_display = [0, 1, 2, num_nodes - 3, num_nodes - 2, num_nodes - 1]
            else:
                nodes_to_display = list(range(num_nodes))

            for i, node_id in enumerate(nodes_to_display):
                weight_sum = numpy.sum(layer.W[:, node_id])
                color_intensity = min(1.0, max(0.0, (weight_sum + 1) / 2))  # Normalize to [0, 1]
                self.texts[layer_idx * len(nodes_to_display) + i].set_text(f'{node_id}\n{weight_sum:.2f}')
                self.texts[layer_idx * len(nodes_to_display) + i].set_color(plt.cm.viridis(color_intensity))

    def record(self, X, y, epochs):
        self._initialize_plot()

        def update(frame):
            self.model.train(X, y, epochs=1)
            self._update_weights()
            return self.texts + self.lines

        ani = animation.FuncAnimation(self.fig, update, frames=epochs, interval=self.interval, blit=True)

        if self.save_path:
            writer = animation.FFMpegWriter(fps=self.fps, bitrate=self.bitrate)
            ani.save(self.save_path, writer=writer)
        else:
            plt.show()
