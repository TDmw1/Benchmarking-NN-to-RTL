import os
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
import nngen as ng
import numpy as np

os.makedirs("quantized/nngen", exist_ok=True)

# ---------------------------------------------------------
# Test Case 1: Ultra-Minimal Single Conv (No Branching)
# ---------------------------------------------------------
class SingleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# ---------------------------------------------------------
# Test Case 2: Minimal Skip Connection (Branching)
# ---------------------------------------------------------
class BranchConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 1x1 conv to match dimensions for the addition
        self.skip = nn.Conv2d(1, 16, kernel_size=1, padding=0)

    def forward(self, x):
        # The input tensor 'x' is consumed by two separate paths
        return self.conv(x) + self.skip(x)

# ---------------------------------------------------------
# Diagnostic Pipeline
# ---------------------------------------------------------
def run_diagnostic(model_class, name):
    print(f"\n--- Running Diagnostic: {name} ---")
    model = model_class()
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = f"quantized/nngen/{name}.onnx"

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=11
    )

    model_proto = onnx.load(onnx_path)
    model_simp, check = simplify(model_proto)
    onnx.save(model_simp, onnx_path)

    try:
        print("Parsing ONNX into NNgen...")
        # CONFIRMED signature: (outputs, placeholders, variables, constants, operators)
        # Position 0 is the real output -- NOT position 2 (that's `variables`).
        outputs, placeholders, variables, constants, operators = ng.from_onnx(
            onnx_path, value_dtypes={"input": ng.int8}
        )

        input_node = placeholders["input"]
        input_name = input_node.name

        print(f"output candidates: {[(k, v.shape) for k, v in outputs.items()]}")
        output_key, output_node = list(outputs.items())[0]
        print(f"selected output: '{output_key}' shape={output_node.shape}")

        print("Running fixed-scale dummy quantization...")
        act_scale_factor = 64
        ng.quantize(
            [output_node],
            {input_name: act_scale_factor},
            {input_name: 0.0},
            {input_name: 1.0},
        )
        print(f"output_node dtype after quantize: {output_node.dtype}")

        print("Running dummy evaluation...")
        dummy_int_input = np.random.randint(-128, 127, size=(1, 1, 28, 28), dtype=np.int64)
        # CONFIRMED call form: keyword-unpacked, not a positional dict.
        eval_out = ng.eval([output_node], **{input_name: dummy_int_input})
        print(f"output shape: {eval_out[0].shape}")

        print(f"SUCCESS: {name} compiled and evaluated cleanly!")

    except Exception as e:
        import traceback
        print(f"FAILED: {name} threw an error: {e}")
        traceback.print_exc()

# Run both tests
run_diagnostic(SingleConv, "diag_single_conv")
run_diagnostic(BranchConv, "diag_branch_conv")