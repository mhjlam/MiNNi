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

import run_example as example


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.FashionMNIST(example.INPUT_DIR, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(example.INPUT_DIR, train=False, download=True, transform=transform)
    return (train, test)


def main():
    train, test = load_data()

    # Update the data loading to match the CNN input shape
    X = train.data.unsqueeze(1).float() / 255.0  # Add channel dimension for grayscale images
    y = train.targets
    Xt = test.data.unsqueeze(1).float() / 255.0
    yt = test.targets

    # Split 10,000 images from the training dataset for evaluation
    X_train, y_train = X[:-10000], y[:-10000]  # First 50,000 images for training
    X_eval, y_eval = X[-10000:], y[-10000:]    # Last 10,000 images for evaluation

    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),

        nn.Flatten(),

        nn.Linear(128 * 7 * 7, 256),
        nn.ReLU(),

        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(128, 10)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    batch_size = 64
    num_epochs = 10

    # Metrics storage
    batch_losses = []
    batch_accuracies = []
    batch_indices = []

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        perm = torch.randperm(X_train.size(0))  # Shuffle training data
        loss_sum = 0.0
        correct = 0
        total = 0

        for i in range(0, X_train.size(0), batch_size):
            batch = perm[i : i + batch_size]
            optimizer.zero_grad()
            outputs = model(X_train[batch])
            loss = criterion(outputs, y_train[batch])
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_train[batch]).sum().item()
            total += batch.size(0)

            # Store batch metrics
            batch_losses.append(loss.item())
            batch_accuracies.append(correct / total * 100)
            batch_indices.append(epoch + i / X_train.size(0))  # Relative batch index within the epoch

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_sum / (X_train.size(0) // batch_size)}, Accuracy: {correct / total * 100:.2f}%")

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        outputs = model(X_eval)
        loss = criterion(outputs, y_eval)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_eval).sum().item() / y_eval.size(0) * 100
        print(f"Evaluation Loss: {loss.item()}, Accuracy: {accuracy:.2f}%")

    # Use 10 random images from the test dataset for predictions
    indices = torch.randperm(Xt.size(0))[:10]  # Select 10 random indices
    test_images = Xt[indices]
    test_labels = yt[indices]

    with torch.no_grad():
        predictions = model(test_images)
        predicted_classes = torch.argmax(predictions, dim=1)

    labels = list(example.FMNIST_LABELS.values())

    # Display predictions
    for i in range(10):
        plt.imshow(test_images[i].squeeze(), cmap="gray")
        color = "red" if test_labels[i] != predicted_classes[i] else "black"
        plt.title(f"True: {labels[test_labels[i]]}, Predicted: {labels[predicted_classes[i]]}", color=color)
        plt.show()

    # Generate confusion matrix
    with torch.no_grad():
        preds = torch.max(model(Xt), 1)[1]
        accuracy = (preds == yt).sum().item() / yt.size(0) * 100
        print(f"Accuracy: {accuracy:.2f}%")

        cm = metrics.confusion_matrix(yt.cpu().numpy(), preds.cpu().numpy())
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(example.OUTPUT_DIR, "pytorch_fmnist-cnn-error.png"))

    # Predicting on custom images
    for img_path in example.PREDICT_IMAGES:
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (28, 28))
        img = torch.tensor((255 - img).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        with torch.no_grad():
            print(f"{img_path} is predicted as {labels[torch.max(model(img), 1)[1].item()]}")

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
    plt.savefig(os.path.join(example.OUTPUT_DIR, "pytorch_fmnist-cnn-loss.png"))
    plt.show()
