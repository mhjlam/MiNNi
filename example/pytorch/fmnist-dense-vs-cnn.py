import os

import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(DIR, os.pardir))
INPUT_DIR = os.path.join(EXAMPLE_DIR, "input")
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "output")
FMNIST_DIR = os.path.join(INPUT_DIR, "fmnist")
UNSEEN_DIR = os.path.join(INPUT_DIR, "example")


# Load MNIST
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(FMNIST_DIR, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

# Augment test data with transformations
test_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Small rotations and translations
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor()])

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(FMNIST_DIR, train=False, download=True, transform=test_transform), 
    batch_size=64)


def plot_confusion_matrix(all_labels, all_predictions, classes, model_name):
    cm = metrics.confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Add text annotations
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the confusion matrix as a PNG file
    filename = os.path.join(OUTPUT_DIR, "pytorch", f"fmnist-{model_name}-error.png")
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix plot saved to {filename}")


def train(model, name):
    print(f"Training model {name}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} complete.")

    # Evaluate
    print(f"Evaluating model {name}...")
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot and save confusion matrix as PNG
    plot_confusion_matrix(all_labels, all_predictions, classes=[str(i) for i in range(10)], model_name=name)


if __name__ == "__main__":
    dense_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10))

    simple_cnn = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
        nn.BatchNorm2d(64), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.BatchNorm2d(128), nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        nn.BatchNorm2d(128), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 3 * 3, 512), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10))

    train(dense_net, name="dense_net")
    train(simple_cnn, name="simple_cnn")
