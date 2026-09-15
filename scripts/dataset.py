import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def get_eval_dataset(batch_size=1, num_samples=1000):
    """
    Returns a deterministic subset of MNIST test data for evaluation.
    - batch_size=1 reflects typical real-time FPGA streaming inference.
    - num_samples limits test runtime while preserving statistical significance.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    
    # Deterministic slice so all models and quantization paths see identical samples
    indices = list(range(min(num_samples, len(test_dataset))))
    subset = torch.utils.data.Subset(test_dataset, indices)
    
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return loader

if __name__ == "__main__":
    loader = get_eval_dataset(batch_size=1, num_samples=10)
    images, labels = next(iter(loader))
    print(f"Dataset loader verified.")
    print(f"Sample tensor shape: {images.shape}, dtype: {images.dtype}")