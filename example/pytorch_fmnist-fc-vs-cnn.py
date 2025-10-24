import os

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import run_example as example


# Load FashionMNIST
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(example.INPUT_DIR, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

# Augment test data with transformations
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(example.INPUT_DIR, train=False, download=True, transform=transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Small rotations and translations
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor()])), 
    batch_size=64)


def plot_confusion_matrix(all_labels, all_predictions, model_name, accuracy):
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=metrics.confusion_matrix(all_labels, all_predictions), 
        display_labels=example.FMNIST_LABELS
    )
    disp.plot(cmap="viridis", xticks_rotation=45)
    plt.gca().xaxis.set_tick_params(labelbottom=True)
    plt.gca().set_xticks(range(len(example.FMNIST_LABELS)))
    plt.gca().set_xticklabels(example.FMNIST_LABELS)
    disp.plot(cmap="viridis", xticks_rotation=45)
    plt.title(f"{model_name} Confusion Matrix\nAccuracy: {accuracy:.2f}%")
    plt.xlabel("")
    plt.ylabel("")

    filename = os.path.join(example.OUTPUT_DIR, f"pytorch_fmnist-vs-{model_name}-error.png")

    plt.tight_layout(pad=2.0)  # Adjust padding for a tighter layout
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
    plot_confusion_matrix(all_labels, all_predictions, model_name=name, accuracy=accuracy)


def main():
    fc = nn.Sequential(nn.Flatten(), 
                       nn.Linear(784, 512), nn.ReLU(), nn.BatchNorm1d(512), 
                       nn.Dropout(0.25),
                       nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), 
                       nn.Dropout(0.5),
                       nn.Linear(512, 10))
    
    cnn = nn.Sequential(nn.Conv2d(1, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2),
                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2),
                        nn.Dropout(0.5),
                        nn.Conv2d(512, 10, 3, padding=1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    train(fc, name="FC")
    train(cnn, name="CNN")
