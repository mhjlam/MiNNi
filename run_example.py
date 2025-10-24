import os
import importlib

FMNIST_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

ROOT = os.path.abspath(os.path.dirname(__file__))
EXAMPLE_DIR = os.path.join(ROOT, "example")
INPUT_DIR = os.path.join(EXAMPLE_DIR, "_input")
OUTPUT_DIR = os.path.join(EXAMPLE_DIR, "_output")

FASHION_MNIST_RAW_DIR = os.path.join(INPUT_DIR, "FashionMNIST", "raw")

FMNIST_LABELS = [
    "Shirt (casual)",
    "Trousers",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt (formal)",
    "Sneaker",
    "Bag",
    "Boot",
]

PREDICT_IMAGES = [
    os.path.join(INPUT_DIR, "bag.png"),
    os.path.join(INPUT_DIR, "boot.png"),
    os.path.join(INPUT_DIR, "coat.png"),
    os.path.join(INPUT_DIR, "dress.png"),
    os.path.join(INPUT_DIR, "jeans.png"),
    os.path.join(INPUT_DIR, "pullover.png"),
    os.path.join(INPUT_DIR, "shirt.png"),
    os.path.join(INPUT_DIR, "sneaker.png"),
    os.path.join(INPUT_DIR, "tshirt.png"),
]


def find_examples():
    examples = []
    for root, _, files in os.walk(EXAMPLE_DIR):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Convert file path to module path
                relative_path = os.path.relpath(os.path.join(root, file), ROOT)
                module_path = relative_path.replace(os.sep, ".").rsplit(".py", 1)[0]
                examples.append(module_path)
    return examples


def run(example_module_path):
    try:
        example_module = importlib.import_module(example_module_path)
        if hasattr(example_module, "main"):
            example_module.main()
        else:
            print(f"Module {example_module_path} has no main() function!")
    except ModuleNotFoundError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    examples = find_examples()

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    os.makedirs(FASHION_MNIST_RAW_DIR, exist_ok=True)

    if not examples:
        print("No examples found!")
        exit(1)

    print("Available examples:")
    for idx, example in enumerate(examples, 1):
        print(f"{idx}. {example}")

    choice = int(input("Select an example to run (number): ")) - 1
    if 0 <= choice < len(examples):
        run(examples[choice])
    else:
        print("Invalid choice!")
