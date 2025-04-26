import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


FMNIST_LABELS = {
    0: "Shirt (casual)",
    1: "Trousers",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt (formal)",
    7: "Sneaker",
    8: "Bag",
    9: "Boot",
}

DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(DIR, os.pardir))
INPUT_DIR = os.path.join(EXAMPLE_DIR, "input")
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "output")
FMNIST_DIR = os.path.join(INPUT_DIR, "fmnist")
UNSEEN_DIR = os.path.join(INPUT_DIR, "example")

PREDICT_IMAGES = [
    os.path.join(UNSEEN_DIR, "bag.png"),
    os.path.join(UNSEEN_DIR, "boot.png"),
    os.path.join(UNSEEN_DIR, "coat.png"),
    os.path.join(UNSEEN_DIR, "dress.png"),
    os.path.join(UNSEEN_DIR, "jeans.png"),
    os.path.join(UNSEEN_DIR, "pullover.png"),
    os.path.join(UNSEEN_DIR, "shirt.png"),
    os.path.join(UNSEEN_DIR, "sneakers.png"),
    os.path.join(UNSEEN_DIR, "tshirt.png"),
]

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.FashionMNIST(FMNIST_DIR, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(FMNIST_DIR, train=False, download=True, transform=transform)
    
    return (train.data.float().view(-1, 28 * 28) / 255.0, train.targets, 
            test.data.float().view(-1, 28 * 28) / 255.0, test.targets)


if __name__ == "__main__":
    X, y, Xt, yt = load_data()

    model = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 128
    num_epochs = 10

    # Metrics storage
    batch_losses = []
    batch_accuracies = []
    batch_indices = []

    model.train()
    for epoch in range(num_epochs):
        perm = torch.randperm(X.size(0))
        loss_sum = 0.0
        correct = 0
        total = 0

        for i in range(0, X.size(0), batch_size):
            batch = perm[i : i + batch_size]
            optimizer.zero_grad()
            outputs = model(X[batch])
            loss = criterion(outputs, y[batch])
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y[batch]).sum().item()
            total += batch.size(0)

            # Store batch metrics
            batch_losses.append(loss.item())
            batch_accuracies.append(correct / total * 100)
            batch_indices.append(epoch + i / X.size(0))  # Relative batch index within the epoch

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_sum / (X.size(0) // batch_size)}, Accuracy: {correct / total * 100:.2f}%")

    model.eval()
    with torch.no_grad():
        preds = torch.max(model(Xt), 1)[1]
        accuracy = (preds == yt).sum().item() / yt.size(0) * 100
        print(f"Accuracy: {accuracy:.2f}%")

        cm = metrics.confusion_matrix(yt.cpu().numpy(), preds.cpu().numpy())
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=FMNIST_LABELS.values())

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(OUTPUT_DIR, "pytorch", "fmnist-dense-error.png"))

    for img_path in PREDICT_IMAGES:
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (28, 28))
        img = torch.tensor((255 - img).astype(np.float32) / 255.0).view(1, -1)
        with torch.no_grad():
            print(f"{img_path} is predicted as {FMNIST_LABELS.values()[torch.max(model(img), 1)[1].item()]}")

    # Plotting metrics
    plt.figure(figsize=(12, 8))

    # Plot loss per batch
    plt.subplot(2, 1, 1)
    plt.plot(batch_indices, batch_losses, label="Loss", color="blue", linewidth=0.5)
    plt.xticks(ticks=range(num_epochs + 1))
    plt.yticks(ticks=np.linspace(0.0, 2.0, num=9))
    plt.ylim(0.0, 2.0)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot accuracy per batch
    plt.subplot(2, 1, 2)
    plt.plot(batch_indices, batch_accuracies, label="Accuracy", color="green", linewidth=0.5)
    plt.xticks(ticks=range(num_epochs+1))
    plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])
    plt.ylim(0, 100)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(os.join.path(OUTPUT_DIR, "pytorch", "fmnist-dense-loss.png"))
    plt.show()
