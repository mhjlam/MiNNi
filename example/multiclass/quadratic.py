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


DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(DIR, os.pardir))
INPUT_DIR = os.path.join(EXAMPLE_DIR, "_input")
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "_output")


def generate_dataset(N):
    numpy.random.seed(420)
    X = []
    while len(X) < N:
        x = numpy.random.rand()  # Generate x in the range [0, 1]
        y = numpy.random.rand()  # Generate y in the range [0, 1]

        # Increase density near corners (0, 0) and (1, 1)
        corner_bias = (
            1.5 * (1 - numpy.sqrt((x - 0) ** 2 + (y - 0) ** 2))) + (
            1.5 * (1 - numpy.sqrt((x - 1) ** 2 + (y - 1) ** 2))
        )
        if numpy.random.rand() < corner_bias:
            X.append([x, y])

    # Add at least one point between x=[0,0.1] but below the y=x^2 boundary line
    X.append([0.05, 0.05**2 - 0.01])  # Example point below the boundary

    X = numpy.array(X)
    boundary_margin = 0.01
    mask = numpy.abs(X[:, 1] - X[:, 0] ** 2) > boundary_margin
    X = X[mask][:N]  # Ensure we still return N points
    y = (X[:, 1] > X[:, 0] ** 2).astype("uint8")  # Label: 1 if y > x^2, else 0
    return X, y


def generate_poisson_dataset(N, width=1, height=1, radius=0.1):
    def poisson_disk_sampling(radius, width, height, k=30):
        cell_size = radius / numpy.sqrt(2)
        grid_width = int(numpy.ceil(width / cell_size))
        grid_height = int(numpy.ceil(height / cell_size))
        grid = -numpy.ones((grid_width, grid_height), dtype=int)

        def grid_coords(point):
            return int(point[0] // cell_size), int(point[1] // cell_size)

        def in_neighborhood(point):
            gx, gy = grid_coords(point)
            for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                    if grid[i, j] != -1:
                        neighbor = points[grid[i, j]]
                        if numpy.linalg.norm(point - neighbor) < radius:
                            return True
            return False

        points = []
        active_list = []

        initial_point = numpy.random.uniform(0, [width, height])
        points.append(initial_point)
        active_list.append(initial_point)
        grid[grid_coords(initial_point)] = 0

        while active_list and len(points) < N:
            idx = numpy.random.randint(len(active_list))
            base_point = active_list[idx]
            found = False
            for _ in range(k):
                angle = numpy.random.uniform(0, 2 * numpy.pi)
                offset = numpy.random.uniform(radius, 2 * radius)
                new_point = base_point + offset * numpy.array([numpy.cos(angle), numpy.sin(angle)])
                
                if (0 <= new_point[0] < width and 0 <= new_point[1] < height and not in_neighborhood(new_point)):
                    points.append(new_point)
                    active_list.append(new_point)
                    grid[grid_coords(new_point)] = len(points) - 1
                    found = True
                    break
            if not found:
                active_list.pop(idx)

        return numpy.array(points)

    numpy.random.seed(420)
    points = poisson_disk_sampling(radius=0.05, width=1, height=1)
    points = points[:N]  # Limit to N points if necessary

    # Filter out points close to the boundary y = x^2
    boundary_margin = 0.01
    mask = numpy.abs(points[:, 1] - points[:, 0] ** 2) > boundary_margin
    points = points[mask][:N]  # Ensure we still return N points
    y = (points[:, 1] > points[:, 0] ** 2).astype("uint8")  # Label: 1 if y > x^2, else 0
    return points, y


def visualize_network(model):
    # Determine the size of the input layer from the first layer's input size
    input_size = model.layers[0].input_size
    input_layer = type("InputLayer", (object,), {"output_size": input_size})()  # Create a mock input layer
    layers = [input_layer] + model.layers  # Include the input layer
    fig, ax = plt.subplots(figsize=(10, 6))

    # Size of the nodes
    node_size = 500

    # Adjust spacing
    layer_spacing = 3
    max_nodes = 6

    # Calculate the total width of the network
    x_offset = fig.get_figwidth() / (len(layers) * layer_spacing)

    for i, layer in enumerate(layers):
        # Get the number of nodes in the current layer
        num_nodes = layer.output_size

        # Calculate the x-coordinate for the current layer, adjusted for centering
        x = i * layer_spacing + x_offset

        # Calculate the y-coordinates for the nodes in the current layer
        y = numpy.linspace(max_nodes / 2 - (max_nodes - num_nodes) / 2,
                        -max_nodes / 2 + (max_nodes - num_nodes) / 2, num_nodes)

        # Handle cases where the number of nodes exceeds 5
        if num_nodes > 5:
            y = numpy.linspace(max_nodes / 2, -max_nodes / 2, 6)

            # Add extra spacing between the first and last 3 nodes
            y = numpy.concatenate([y[:3] + 0.5, y[-3:] - 0.5])

        # Plot the nodes with the updated size
        ax.scatter([x] * len(y), y, s=node_size, color="black", edgecolors="gray", linewidth=2, zorder=3)

        # Add labels for the nodes
        for j, y_coord in enumerate(y):
            node_label = (f"{j}" if num_nodes <= 5 else (f"{j}" if j < 3 else f"{num_nodes - len(y) + j}"))
            fontsize = 10
            ax.text(x, y_coord - 0.04 * (10 / fontsize), node_label, 
                    fontsize=fontsize, ha="center", va="center", color="white", zorder=4)

        # Draw connections from the previous layer to the current layer
        if i > 0:
            prev_layer = layers[i - 1]
            prev_num_nodes = prev_layer.output_size
            prev_x = (i - 1) * layer_spacing + x_offset
            prev_y = numpy.linspace(max_nodes / 2 - (max_nodes - prev_num_nodes) / 2,
                                 -max_nodes / 2 + (max_nodes - prev_num_nodes) / 2, prev_num_nodes)

            if prev_num_nodes > 5:
                prev_y = numpy.linspace(max_nodes / 2, -max_nodes / 2, 6)
                prev_y = numpy.concatenate([prev_y[:3] + 0.5, prev_y[-3:] - 0.5])

            for j, prev_y_coord in enumerate(prev_y):
                for k, curr_y_coord in enumerate(y):
                    ax.plot([prev_x, x], [prev_y_coord, curr_y_coord], color="gray", zorder=1)
                    mid_x = (prev_x + x) / 2
                    mid_y = (prev_y_coord + curr_y_coord) / 2

                    # Change the weight label color to black
                    if (i == 1 or i == len(layers) - 1):  # Show labels only for input and output layers
                        weight = layer.forward(numpy.eye(layer.input_size))[j, k]
                        ax.text(mid_x, mid_y, f"{weight:.2f}", fontsize=8, color="black", zorder=2)

    # Set plot limits and labels
    ax.set_xlim(-1, len(layers) * layer_spacing)
    ax.set_ylim(-max_nodes / 2 - 1, max_nodes / 2 + 1)
    ax.axis("off")
    plt.show()


def train_network(model, X, y, epochs=100):
    input_size = model.layers[0].input_size
    input_layer = type("InputLayer", (object,), {"output_size": input_size})()
    layers = [input_layer] + model.layers  # Include the input layer
    fig, ax = plt.subplots(figsize=(10, 6))

    node_size = 500
    layer_spacing = 3
    max_nodes = max(16, input_size)
    x_offset = fig.get_figwidth() / (len(layers) * layer_spacing)

    # Precompute node positions
    node_positions = []
    for i, layer in enumerate(layers):
        num_nodes = layer.output_size
        x_coords = i * layer_spacing + x_offset
        y_coords = numpy.linspace(max_nodes / 2 - (max_nodes - num_nodes) / 2, 
                               -max_nodes / 2 + (max_nodes - num_nodes) / 2, num_nodes)
        if num_nodes > 5:
            y_coords = numpy.linspace(max_nodes / 2, -max_nodes / 2, max_nodes)
            if max_nodes > 16:
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
            if i == 0:
                # Input layer: Red nodes influenced by values
                if i < len(normalized_values):
                    colors = [plt.cm.Reds(value) for value in normalized_values[i].flatten()[: len(y_coords)]]
                else:
                    colors = ["red"] * len(y_coords)  # Default red for uninitialized layers
                edge_colors = ["darkred"] * len(y_coords)
            elif i == len(node_positions) - 1:
                # Output layer: Green nodes influenced by values
                if i < len(normalized_values):
                    colors = [plt.cm.Greens(value) for value in normalized_values[i].flatten()[: len(y_coords)]]
                else:
                    colors = ["green"] * len(y_coords)  # Default green for uninitialized layers
                edge_colors = ["darkgreen"] * len(y_coords)
            else:
                # Hidden layers: Deeper blue nodes influenced by values
                if i < len(normalized_values):
                    colors = [plt.cm.Blues(value) for value in normalized_values[i].flatten()[: len(y_coords)]]
                else:
                    colors = ["blue"] * len(y_coords)  # Default blue for uninitialized layers
                edge_colors = ["darkblue"] * len(y_coords)

            # Draw nodes with colors based on their values
            ax.scatter([x_coords] * len(y_coords), y_coords, 
                       s=node_size, c=colors, edgecolors=edge_colors, linewidth=2, zorder=3)

            # Draw connections
            if i > 0:
                prev_x, prev_y_coords = node_positions[i - 1]
                for j, prev_y_coord in enumerate(prev_y_coords):
                    for k, curr_y_coord in enumerate(y_coords):
                        ax.plot([prev_x, x_coords], [prev_y_coord, curr_y_coord], color="gray", zorder=1)

    ani = animation.FuncAnimation(fig, update, frames=epochs, interval=200, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, "multiclass", "quadratic-net.mp4"), writer="ffmpeg", fps=30)


def train_plot(model, X, y, epochs=100):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate a grid of points to evaluate the decision boundary
    x_min, x_max = 0, X[:, 0].max()
    y_min, y_max = 0, X[:, 1].max()
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 200), numpy.linspace(y_min, y_max, 200))

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Epoch {frame + 1}/{epochs}")

        # Train the model for one epoch
        model.train(X, y, epochs=1)

        # Predict on the grid points
        grid_points = numpy.c_[xx.ravel(), yy.ravel()]
        predictions = model.predict(grid_points)
        if predictions.ndim == 1:  # Handle 1-dimensional predictions
            predictions = predictions.reshape(-1, 1)
        predictions = predictions[:, 0].reshape(xx.shape)  # Adjust indexing for class 1

        # Plot the decision boundary
        ax.contourf(xx, yy, predictions, levels=numpy.linspace(0, 1, 50), cmap="RdBu", alpha=0.6)
        ax.contour(xx, yy, predictions, levels=[0.5], colors="black", linewidths=1)

        # Plot the original points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="white", s=20, zorder=3)

        # Plot the true decision boundary (y = x^2)
        x_vals = numpy.linspace(x_min, x_max, 200)
        ax.plot(x_vals, x_vals**2, color="black", label="True Boundary")
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=epochs, interval=50, repeat=False)
    ani.save(os.path.join(OUTPUT_DIR, "multiclass", "quadratic-fit.mp4"), writer="ffmpeg", fps=30)


if __name__ == "__main__":
    print("\nClassification (random points: above/below y=x^2)")

    samples = 1000  # Number of points

    X, y = generate_dataset(N=samples)
    Xt, yt = generate_dataset(N=samples)

    model = minni.model.Model(loss=minni.loss.CrossEntropy(), 
                              optimizer=minni.optimizer.Adam(eta=0.05, beta=0.000005))
    model.add(minni.layer.Dense(2, 64, activator=minni.activator.Rectifier(), 
                                regularizer=minni.regularizer.Ridge(0.05)))
    model.add(minni.layer.Dense(64, 64, activator=minni.activator.Rectifier(), 
                                regularizer=minni.regularizer.Ridge(0.05)))
    model.add(minni.layer.Dense(64, 2, activator=minni.activator.Softmax()))

    train_network(model, X, y, epochs=500)
