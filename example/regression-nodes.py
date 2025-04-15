import os
import sys
import numpy
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import minni
import minni.loss
import minni.layer
import minni.model
import minni.activator
import minni.optimizer
import minni.visualizer
import minni.initializer

minni.init()


def generate_sine_dataset(N=1000):
    X = numpy.linspace(-1, 1, N).reshape(-1, 1)
    y = numpy.sin(X * X).reshape(-1, 1)
    return X, y


def visualize_network(model):
    """
    Visualize the neural network structure with matplotlib, including the input layer.
    """
    input_size = model.layers[0].input_size  # Determine the size of the input layer from the first layer's input size
    input_layer = type('InputLayer', (object,), {'output_size': input_size})()  # Create a mock input layer
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
        y = np.linspace(max_nodes / 2 - (max_nodes - num_nodes) / 2, -max_nodes / 2 + (max_nodes - num_nodes) / 2, num_nodes)
        
        # Handle cases where the number of nodes exceeds 5
        if num_nodes > 5:
            y = np.linspace(max_nodes / 2, -max_nodes / 2, 6)
            y = np.concatenate([y[:3] + 0.5, y[-3:] - 0.5])  # Add extra spacing between the first and last 3 nodes
        
        # Plot the nodes with the updated size
        ax.scatter([x] * len(y), y, s=node_size, color='black', edgecolors='gray', linewidth=2, zorder=3)
        
        # Add labels for the nodes
        for j, y_coord in enumerate(y):
            node_label = f'{j}' if num_nodes <= 5 else (f'{j}' if j < 3 else f'{num_nodes - len(y) + j}')
            fontsize = 10
            ax.text(x, y_coord - 0.04 * (10 / fontsize), node_label, fontsize=fontsize, ha='center', va='center', color='white', zorder=4)
        
        # Draw connections from the previous layer to the current layer
        if i > 0:
            prev_layer = layers[i - 1]
            prev_num_nodes = prev_layer.output_size
            prev_x = (i - 1) * layer_spacing + x_offset
            prev_y = np.linspace(max_nodes / 2 - (max_nodes - prev_num_nodes) / 2, -max_nodes / 2 + (max_nodes - prev_num_nodes) / 2, prev_num_nodes)
            
            if prev_num_nodes > 5:
                prev_y = np.linspace(max_nodes / 2, -max_nodes / 2, 6)
                prev_y = np.concatenate([prev_y[:3] + 0.5, prev_y[-3:] - 0.5])
            
            for j, prev_y_coord in enumerate(prev_y):
                for k, curr_y_coord in enumerate(y):
                    ax.plot([prev_x, x], [prev_y_coord, curr_y_coord], color='gray', zorder=1)
                    mid_x = (prev_x + x) / 2
                    mid_y = (prev_y_coord + curr_y_coord) / 2
                    
                    # Change the weight label color to black
                    if i == 1 or i == len(layers) - 1:  # Show labels only for input and output layers
                        weight = layer.forward(np.eye(layer.input_size))[j, k]
                        ax.text(mid_x, mid_y, f'{weight:.2f}', fontsize=8, color='black', zorder=2)
    
    # Set plot limits and labels
    ax.set_xlim(-1, len(layers) * layer_spacing)
    ax.set_ylim(-max_nodes / 2 - 1, max_nodes / 2 + 1)
    ax.axis('off')
    plt.title('Neural Network Visualization')
    plt.show()


if __name__ == '__main__':
    print('Regression (sine)')
    
    X, y = generate_sine_dataset()

    model = minni.model.Model(loss=minni.loss.MeanSquaredError(),
                              optimizer=minni.optimizer.Adam(eta=0.005, beta=0.001),
                              metric=minni.Metric.REGRESSION)
    
    rand_scaled = minni.initializer.Random(scaler=0.1)
    model.add(minni.layer.Dense(1, 32, rand_scaled, minni.activator.Rectifier()))
    model.add(minni.layer.Dense(32, 32, rand_scaled, minni.activator.Rectifier()))
    model.add(minni.layer.Dense(32, 1, rand_scaled, minni.activator.Linear()))

    model.train(X, y, epochs=1000)
    
    # Visualize the network
    visualize_network(model)
