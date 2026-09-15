import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.makedirs("models/mlp_small", exist_ok=True)
os.makedirs("models/cnn_lenet", exist_ok=True)
os.makedirs("models/cnn_skip", exist_ok=True)
os.makedirs("models/cnn_unusual_op", exist_ok=True)

# 1. Models Definition
class MLPSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CNNLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CNNSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        identity = self.conv1(x)
        out = F.relu(identity)
        out = self.conv2(out)
        out = F.relu(out + identity)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

class CNNUnusualOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv_grouped = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv_grouped(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 2. Quick Training Routine (1 Epoch is enough for >92-96% on MNIST)
def quick_train(model, train_loader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Load MNIST Training Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

models = {
    "mlp_small": MLPSmall(),
    "cnn_lenet": CNNLeNet(),
    "cnn_skip": CNNSkip(),
    "cnn_unusual_op": CNNUnusualOp()
}

dummy_input = torch.randn(1, 1, 28, 28)

for name, model in models.items():
    print(f"Training {name} on MNIST (1 epoch)...")
    quick_train(model, train_loader, epochs=1)
    
    model.eval()
    export_path = f"models/{name}/model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None
    )
    print(f"Exported trained: {export_path}")