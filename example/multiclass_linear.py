import os

import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import minni
import minni.loss
import minni.layer
import minni.model
import minni.activator
import minni.optimizer
import minni.regularizer

import run_example as example


def generate_dataset(N):
    numpy.random.seed(420)
    X = numpy.random.rand(N, 2) * 2 - 1  # Generate points in the range [-1, 1]
    y = (X[:, 1] > X[:, 0]).astype("uint8")  # Label: 1 if y > x, else 0
    return X, y


def train(model, X, y, epochs=100):
    input_size = model.layers[0].input_size
    input_layer = type("InputLayer", (object,), {"output_size": input_size})()
    layers = [input_layer] + model.layers
    fig, ax = plt.subplots(figsize=(10, 6))

    node_size = 500
    layer_spacing = 3
    max_nodes = max(16, input_size)
    x_offset = fig.get_figwidth() / (len(layers) * layer_spacing)

    # Precompute node positions
    node_positions = []
    for i, layer in enumerate(layers):
        if not isinstance(layer, minni.layer.Dense):
            continue

        num_nodes = layer.output_size
        x_coords = i * layer_spacing + x_offset
        y_coords = numpy.linspace(max_nodes / 2 - (max_nodes - num_nodes) / 2,
                                  -max_nodes / 2 + (max_nodes - num_nodes) / 2, num_nodes)
        
        if num_nodes > 5:
            y_coords = numpy.linspace(max_nodes / 2, -max_nodes / 2, max_nodes)
            y_coords = numpy.concatenate([y_coords[: max_nodes // 2] + 0.5, y_coords[-max_nodes // 2 :] - 0.5])
        node_positions.append((x_coords, y_coords))

    def update(_):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(-1, len(layers) * layer_spacing)
        ax.set_ylim(-max_nodes / 2 - 1, max_nodes / 2 + 1)

        # Train the model for one epoch
        model.train(X, y, epochs=1)

        # Forward pass to get node values
        node_values = [X]  # Start with input values
        for layer in model.layers:
            node_values.append(layer.forward(node_values[-1]))

        # Normalize node values to range [0, 1] for color mapping
        normalized_values = [
            numpy.clip((values - numpy.min(values)) / (numpy.max(values) - numpy.min(values) + 1e-8), 0, 1) 
            for values in node_values
        ]

        # Draw nodes and connections
        for i, (x_coords, y_coords) in enumerate(node_positions):
            # Map node values to grayscale colors (0 = black, 1 = white)
            if i < len(normalized_values):
                colors = [str(1 - value) for value in normalized_values[i].flatten()[: len(y_coords)]]
            else:
                colors = ["0.5"] * len(y_coords)  # Default gray for uninitialized layers

            # Draw nodes with colors based on their values
            ax.scatter([x_coords] * len(y_coords), y_coords, 
                       s=node_size, c=colors, edgecolors="gray", linewidth=2, zorder=3)

            # Draw connections
            if i > 0:
                prev_x, prev_y_coords = node_positions[i - 1]
                for j, prev_y_coord in enumerate(prev_y_coords):
                    for k, curr_y_coord in enumerate(y_coords):
                        ax.plot([prev_x, x_coords], [prev_y_coord, curr_y_coord], color="gray", zorder=1)

    ani = animation.FuncAnimation(fig, update, frames=epochs, interval=200, repeat=False)
    ani.save(os.path.join(example.OUTPUT_DIR, "multiclass_linear-net.mp4"), writer="ffmpeg", fps=30)


def main():
    print("\nClassification (random points: left/right of y=x)")

    samples = 1000  # Number of points

    X, y = generate_dataset(N=samples)

    model = minni.model.Model(loss=minni.loss.CrossEntropy(), optimizer=minni.optimizer.Adam(eta=0.05, beta=0.00005))
    model.add(minni.layer.Dense(2, 32, activator=minni.activator.Rectifier(), regularizer=minni.regularizer.Ridge(0.0005)))
    model.add(minni.layer.Dense(32, 32, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(32, 2, activator=minni.activator.Softmax()))

    # Visualize the network during training
    train(model, X, y, epochs=100)
