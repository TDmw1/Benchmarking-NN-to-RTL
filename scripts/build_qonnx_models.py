import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import brevitas.nn as qnn
from brevitas.export import export_qonnx

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data_tables", "accuracy_fidelity.csv")
os.makedirs("quantized/qonnx", exist_ok=True)

BIT_WIDTH = 8

# ---------------------------------------------------------------------------
# Brevitas-quantized model definitions.
# Structurally identical to build_models.py's FP32 versions -- only the
# layer types change (nn.Conv2d -> qnn.QuantConv2d, etc.) and a QuantIdentity
# is added at the input so the very first tensor entering the graph is
# quantized too, not just weights/activations inside the network.
# Bias left unquantized (float) -- common, low-risk default; bias has a much
# smaller dynamic range and this avoids extra bias-quantizer configuration.
# ---------------------------------------------------------------------------

class QuantMLPSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=BIT_WIDTH)
        self.fc1 = qnn.QuantLinear(28 * 28, 64, bias=True, weight_bit_width=BIT_WIDTH)
        self.relu1 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.fc2 = qnn.QuantLinear(64, 32, bias=True, weight_bit_width=BIT_WIDTH)
        self.relu2 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.fc3 = qnn.QuantLinear(32, 10, bias=True, weight_bit_width=BIT_WIDTH)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.quant_inp(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class QuantCNNLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=BIT_WIDTH)
        self.conv1 = qnn.QuantConv2d(1, 6, kernel_size=5, bias=True, weight_bit_width=BIT_WIDTH)
        self.relu1 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = qnn.QuantConv2d(6, 16, kernel_size=5, bias=True, weight_bit_width=BIT_WIDTH)
        self.relu2 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.fc1 = qnn.QuantLinear(16 * 4 * 4, 64, bias=True, weight_bit_width=BIT_WIDTH)
        self.relu3 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.fc2 = qnn.QuantLinear(64, 10, bias=True, weight_bit_width=BIT_WIDTH)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


class QuantCNNSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=BIT_WIDTH)
        self.conv1 = qnn.QuantConv2d(1, 8, kernel_size=3, padding=1, bias=True, weight_bit_width=BIT_WIDTH)
        self.relu1 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.conv2 = qnn.QuantConv2d(8, 8, kernel_size=3, padding=1, bias=True, weight_bit_width=BIT_WIDTH)
        # Requantize before the add -- both branches need a matching quant
        # representation for the merge to trace cleanly through export.
        self.requant = qnn.QuantIdentity(bit_width=BIT_WIDTH)
        self.relu2 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = qnn.QuantLinear(8 * 14 * 14, 10, bias=True, weight_bit_width=BIT_WIDTH)

    def forward(self, x):
        x = self.quant_inp(x)
        identity = self.conv1(x)
        identity_q = self.requant(identity)
        out = self.relu1(identity)
        out = self.conv2(out)
        out = self.relu2(out + identity_q)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class QuantCNNUnusualOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=BIT_WIDTH)
        self.conv1 = qnn.QuantConv2d(1, 8, kernel_size=3, padding=1, bias=True, weight_bit_width=BIT_WIDTH)
        self.relu1 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.conv_grouped = qnn.QuantConv2d(
            8, 8, kernel_size=3, padding=1, groups=4, bias=True, weight_bit_width=BIT_WIDTH
        )
        self.relu2 = qnn.QuantReLU(bit_width=BIT_WIDTH)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = qnn.QuantLinear(8 * 14 * 14, 10, bias=True, weight_bit_width=BIT_WIDTH)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv_grouped(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training (lightweight QAT -- no pretrained checkpoint exists to calibrate
# from, since build_models.py only exports ONNX and never saves a .pth).
# Same routine build_models.py already uses for the FP32 baselines.
# ---------------------------------------------------------------------------

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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

models = {
    "mlp_small": QuantMLPSmall(),
    "cnn_lenet": QuantCNNLeNet(),
    "cnn_skip": QuantCNNSkip(),
    "cnn_unusual_op": QuantCNNUnusualOp(),
}

dummy_input = torch.randn(1, 1, 28, 28)

for model_name, model in models.items():
    print(f"\n--- Training and exporting {model_name} (Brevitas QAT, {BIT_WIDTH}-bit) ---")
    quick_train(model, train_loader, epochs=1)
    model.eval()

    # Evaluate directly on the Brevitas model in PyTorch -- the exported
    # QONNX file uses a custom `Quant` op domain that plain onnxruntime
    # cannot execute without the qonnx package's custom op registration.
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in eval_loader:
            if total >= 1000:
                break
            out = model(images)
            pred = torch.argmax(out, dim=1)
            correct += int(pred.item() == labels.item())
            total += 1
    q_acc = round((correct / total) * 100.0, 2)
    print(f"[QONNX(native) - {model_name}] Quantized Accuracy: {q_acc:.2f}%")

    # Preserve the exact same output path as the original quantize_qonnx.py
    dst_onnx = f"quantized/qonnx/{model_name}_int8.onnx"
    export_qonnx(model, args=dummy_input, export_path=dst_onnx)
    print(f"Exported native QONNX: {dst_onnx}")

    # Preserve the exact same accuracy_fidelity.csv update logic
    df = pd.read_csv(CSV_PATH)
    mask = (df["model_name"] == model_name) & (df["quant_tool"] == "qonnx")
    baseline = df.loc[mask, "fp32_baseline_accuracy"].values[0]
    df.loc[mask, "quantized_accuracy"] = q_acc
    df.loc[mask, "accuracy_delta"] = round(q_acc - baseline, 2)
    df.to_csv(CSV_PATH, index=False)

print("\nPath A (native QONNX via Brevitas) completed and recorded.")