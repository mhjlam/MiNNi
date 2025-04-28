import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import minni
import minni.loss
import minni.layer
import minni.model
import minni.activator
import minni.optimizer

import run_example as example


def load_data_pytorch():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.FashionMNIST(example.INPUT_DIR, train=True, download=True, transform=transform)
    return train.data.float().view(-1, 28 * 28) / 255.0, train.targets


def preprocess_mnist_fashion_dataset():
    X, y = load_data_pytorch()
    X = (X.numpy() - 0.5) * 2  # Scale to [-1, 1]
    y = y.numpy()
    return X, y


def train_minni_model(X, y, num_epochs, batch_size):
    model = minni.model.Model(loss=minni.loss.CrossEntropy(), optimizer=minni.optimizer.Adam(eta=0.001))
    model.add(minni.layer.Dense(X.shape[1], 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(128, 128, activator=minni.activator.Rectifier()))
    model.add(minni.layer.Dense(128, 10, activator=minni.activator.Softmax()))

    num_batches = len(X) // batch_size
    batch_losses = []
    batch_accuracies = []
    batch_indices = []

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            y_batch = y_batch.astype(np.int64)

            forward_output = model.forward(X_batch)
            y_pred = forward_output[0] if isinstance(forward_output, tuple) else forward_output

            batch_loss, _ = model.loss(y_pred, y_batch, model.layers)
            epoch_loss += batch_loss

            model.backward(y_pred, y_batch)
            model.optimizer.optimize(model.layers)

            correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
            total += y_batch.size

            batch_losses.append(batch_loss)
            batch_accuracies.append(correct / total * 100)
            batch_indices.append(epoch + batch / num_batches)

        avg_loss = epoch_loss / num_batches
        avg_accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.2f}%")

    return batch_losses, batch_accuracies, batch_indices


def train_pytorch_model(X, y, num_epochs, batch_size):
    model = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    batch_losses = []
    batch_accuracies = []
    batch_indices = []

    for epoch in range(num_epochs):
        perm = torch.randperm(X.size(0))
        correct = 0
        total = 0
        epoch_loss = 0

        for i in range(0, X.size(0), batch_size):
            batch = perm[i : i + batch_size]
            optimizer.zero_grad()
            outputs = model(X[batch])
            loss = criterion(outputs, y[batch])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y[batch]).sum().item()
            total += batch.size(0)

            batch_losses.append(loss.item())
            batch_accuracies.append(correct / total * 100)
            batch_indices.append(epoch + i / X.size(0))

        avg_loss = epoch_loss / (X.size(0) // batch_size)
        avg_accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.2f}%")

    return batch_losses, batch_accuracies, batch_indices


def main():
    # Load and preprocess data
    X_minni, y_minni = preprocess_mnist_fashion_dataset()
    X_pytorch, y_pytorch = load_data_pytorch()

    # Training parameters
    num_epochs = 10
    batch_size = 128

    # Train MiNNi model
    print('Training MiNNi model...')
    minni_losses, minni_accuracies, minni_indices = train_minni_model(X_minni, y_minni, num_epochs, batch_size)

    # Train PyTorch model
    print('\nTraining PyTorch model...')
    pytorch_losses, pytorch_accuracies, pytorch_indices = train_pytorch_model(
        X_pytorch, y_pytorch, num_epochs, batch_size)

    # Plot combined metrics
    plt.figure(figsize=(12, 8))

    # Plot loss per batch
    plt.subplot(2, 1, 1)
    plt.plot(minni_indices, minni_losses, label="MiNNi", color="red", linewidth=0.25)
    plt.plot(pytorch_indices, pytorch_losses, label="PyTorch", color="black", linewidth=0.25)
    plt.xticks(ticks=range(num_epochs + 1))
    plt.yticks(ticks=np.linspace(0.0, 2.0, num=9))
    plt.ylim(0.0, 2.0)
    plt.title("Loss Comparison")
    plt.xlabel("Epoch")
    plt.legend(handles=[
        lines.Line2D([0], [0], color="red", linewidth=2, label="MiNNi"),
        lines.Line2D([0], [0], color="black", linewidth=2, label="PyTorch")
    ], loc="upper right")

    # Show final accuracy values
    minni_offset = 0.1 if minni_losses[-1] <= pytorch_losses[-1] else -0.1
    pytorch_offset = -0.1 if minni_losses[-1] <= pytorch_losses[-1] else 0.1
    plt.text(minni_indices[-1], minni_losses[-1] + minni_offset, f"{minni_losses[-1]:.2f}", color="red", fontsize=8)
    plt.text(pytorch_indices[-1], pytorch_losses[-1] + pytorch_offset, f"{pytorch_losses[-1]:.2f}", color="black", fontsize=8)

    # Plot accuracy per batch
    plt.subplot(2, 1, 2)
    plt.plot(minni_indices, minni_accuracies, label="MiNNi", color="red", linewidth=0.5)
    plt.plot(pytorch_indices, pytorch_accuracies, label="PyTorch", color="black", linewidth=0.5)
    plt.xticks(ticks=range(num_epochs + 1))
    plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])
    plt.ylim(0, 100)
    plt.title("Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.legend(handles=[
        lines.Line2D([0], [0], color="red", linewidth=2, label="MiNNi"),
        lines.Line2D([0], [0], color="black", linewidth=2, label="PyTorch")
    ], loc="lower right")

    # Show final accuracy values
    minni_offset = -2 if minni_accuracies[-1] <= pytorch_accuracies[-1] else 2
    pytorch_offset = 2 if minni_accuracies[-1] <= pytorch_accuracies[-1] else -2
    plt.text(minni_indices[-1], minni_accuracies[-1] + minni_offset, f"{minni_accuracies[-1]:.2f}%", color="red", fontsize=8)
    plt.text(pytorch_indices[-1], pytorch_accuracies[-1] + pytorch_offset, f"{pytorch_accuracies[-1]:.2f}%", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(example.OUTPUT_DIR, "pytorch_fmnist-vs-minni.png"))
    plt.show()
