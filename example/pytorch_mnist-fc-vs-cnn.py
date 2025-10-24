import torch
import torchvision
import torchvision.transforms as tf

import run_example as example

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(example.INPUT_DIR, train=True, download=True, transform=tf.ToTensor()),
    batch_size=64, shuffle=True
)

# Augment test data with transformations
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(example.INPUT_DIR, train=False, download=True, transform=tf.Compose([
        tf.RandomAffine(degrees=10, translate=(0.1, 0.1)), 
        tf.RandomHorizontalFlip(), tf.ToTensor()])),
    batch_size=64
)


def train(model, epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')


def main():
    # Fully connected model (requires flattened input)
    fc = torch.nn.Sequential(
        torch.nn.Flatten(),  # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        torch.nn.Linear(784, 512), torch.nn.ReLU(), torch.nn.BatchNorm1d(512), torch.nn.Dropout(0.25),
        torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.BatchNorm1d(512), torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 10)
    )
    
    # Convolutional neural network (works with original image shape)
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(32), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(64), torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.5),
        torch.nn.Conv2d(64, 10, 3, padding=1), torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    )
    
    train(fc)
    train(cnn)


if __name__ == "__main__":
    main()
