import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def visualize_network_animation(weights, num_frames=50):
    """
    Animate a simple feedforward neural network with changing weights.
    
    Args:
        weights (list of numpy arrays): A list where each element is a 2D numpy array
                                        representing the weights between two layers.
                                        Shape of each array: (nodes_in_current_layer, nodes_in_next_layer)
        num_frames (int): Number of frames in the animation.
    """
    num_layers = len(weights) + 1  # Number of layers (including input and output)
    max_nodes = max(max(w.shape) for w in weights)  # Maximum number of nodes in any layer

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # Turn off the axes
    ax.set_aspect('equal')  # Ensure circles are perfectly circular

    # Define layer positions
    layer_positions = np.linspace(0, 1, num_layers)

    # Store node positions for reuse
    node_positions_per_layer = []

    # Define colors
    node_color = "#4CAF50"  # Green for nodes
    weight_color = "#90A4AE"  # Gray for weights
    text_color = "#37474F"  # Dark gray for text

    # Plot each layer (nodes only, no weights yet)
    for layer_idx in range(num_layers):
        if layer_idx == 0:
            num_nodes = weights[0].shape[0]  # Input layer nodes
        elif layer_idx == num_layers - 1:
            num_nodes = weights[-1].shape[1]  # Output layer nodes
        else:
            num_nodes = weights[layer_idx - 1].shape[1]  # Hidden layer nodes

        # Node positions in the current layer
        node_positions = np.linspace(0.5 - num_nodes / (2 * max_nodes), 
                                      0.5 + num_nodes / (2 * max_nodes), 
                                      num_nodes)
        node_positions_per_layer.append(node_positions)

        # Plot nodes
        for y in node_positions:
            ax.add_artist(plt.Circle((layer_positions[layer_idx], y), 0.03, 
                                      color=node_color, edgecolor='black', linewidth=1.5, alpha=0.9, zorder=2))

    # Function to update the weights during animation
    def update(frame):
        ax.clear()
        ax.axis('off')
        ax.set_aspect('equal')  # Ensure circles remain perfectly circular

        # Update title with the current frame
        ax.set_title(f"Neural Network Weights (Frame {frame + 1}/{num_frames})", color=text_color, fontsize=14)

        # Simulate weight changes (e.g., random updates)
        updated_weights = [w + np.random.uniform(-0.1, 0.1, w.shape) for w in weights]

        # Redraw nodes and weights
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                num_nodes = weights[0].shape[0]
            elif layer_idx == num_layers - 1:
                num_nodes = weights[-1].shape[1]
            else:
                num_nodes = weights[layer_idx - 1].shape[1]

            node_positions = node_positions_per_layer[layer_idx]

            # Plot nodes
            for y in node_positions:
                ax.add_artist(plt.Circle((layer_positions[layer_idx], y), 0.025, 
                                          color=node_color, edgecolor='black', linewidth=2, zorder=2))

            # Plot weights
            if layer_idx < num_layers - 1:
                next_node_positions = node_positions_per_layer[layer_idx + 1]
                for i, y1 in enumerate(node_positions):
                    for j, y2 in enumerate(next_node_positions):
                        weight = updated_weights[layer_idx][i, j]
                        ax.plot([layer_positions[layer_idx], layer_positions[layer_idx + 1]], [y1, y2], 
                                color=weight_color, alpha=0.7, linewidth=1.5, zorder=1)
                        ax.text((layer_positions[layer_idx] + layer_positions[layer_idx + 1]) / 2, 
                                (y1 + y2) / 2, f"{weight:.2f}", color=text_color, fontsize=8, ha='center', va='center')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

    # Show the animation
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example weights for a simple 3-layer neural network
    weights = [
        np.random.uniform(-1, 1, (3, 4)),  # Weights between input layer (3 nodes) and hidden layer (4 nodes)
        np.random.uniform(-1, 1, (4, 2))   # Weights between hidden layer (4 nodes) and output layer (2 nodes)
    ]

    visualize_network_animation(weights, num_frames=50)