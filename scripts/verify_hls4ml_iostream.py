import os
import numpy as np
import torch
from torchvision import datasets, transforms

import hls4ml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.core.onnx_exec import execute_onnx

MODEL_NAME = "cnn_skip"
INT8_DIR = "quantized/qonnx"
HLS_PROJECT_DIR = "hls_projects/hls4ml_int8_iostream_test/cnn_skip"
ONNX_PATH = os.path.join(INT8_DIR, f"{MODEL_NAME}_int8.onnx")

# --- 1. Pull one real test sample (match whatever normalization build_qonnx_models.py used) ---
# NOTE: adjust mean/std here if quick_train() used different values than 0.1307/0.3081
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
x_sample, y_label = test_set[0]  # first test image; swap index to try others
x_np = x_sample.numpy().astype(np.float32)  # shape (1, 28, 28)

print(f"Test sample true label: {y_label}")

# --- 2. Reference output: execute the REAL QONNX graph (handles the custom Quant op) ---
qonnx_model = ModelWrapper(ONNX_PATH)
qonnx_model = cleanup_model(qonnx_model)

input_name = qonnx_model.graph.input[0].name
# QONNX graph expects the same input shape the model was exported with, e.g. (1, 1, 28, 28)
ref_input = x_np.reshape(1, 1, 28, 28)

input_dict = {input_name: ref_input}
ref_output_dict = execute_onnx(qonnx_model, input_dict)
ref_output_name = qonnx_model.graph.output[0].name
y_ref = ref_output_dict[ref_output_name].flatten()

print(f"QONNX reference output: {y_ref}")
print(f"QONNX reference argmax: {np.argmax(y_ref)}")

# --- 3. hls4ml output: rebuild the same transformed model, compile, predict ---
hls_model_wrapper = ModelWrapper(ONNX_PATH)
hls_model_wrapper = cleanup_model(hls_model_wrapper)
hls_model_wrapper = hls_model_wrapper.transform(ConvertToChannelsLastAndClean())
hls_model_wrapper = hls_model_wrapper.transform(GemmToMatMul())
hls_model_wrapper = cleanup_model(hls_model_wrapper)

config = hls4ml.utils.config_from_onnx_model(
    hls_model_wrapper,
    granularity='name',
    backend='Vivado'
)
config['Model']['ReuseFactor'] = 1

# Fix confirmed on mlp_small (12.0% -> 100.0%) and cnn_lenet (11.0% -> 100.0%) argmax agreement:
# hls4ml's default 'model_default_t' (ap_fixed<16,6>, 10 fractional bits, resolution ~0.001)
# silently rounds real QONNX quant scales smaller than that to zero in the fused Add/ApplyAlpha
# layers, zeroing out the entire weighted contribution and leaving only the bias term. Testing
# here whether the same override also resolves cnn_skip's Add-node (skip-connection merge) case.
for layer_name in list(config['LayerName'].keys()):
    if (layer_name.startswith('Add_') or layer_name.startswith('Quant_')) and '_param' not in layer_name:
        config['LayerName'][layer_name].setdefault('Precision', {})
        config['LayerName'][layer_name]['Precision']['scale'] = 'ap_fixed<32,16>'
        config['LayerName'][layer_name]['Precision']['bias'] = 'ap_fixed<32,16>'

hls_model = hls4ml.converters.convert_from_onnx_model(
    hls_model_wrapper,
    hls_config=config,
    output_dir=HLS_PROJECT_DIR,
    part='xczu9eg-ffvb1156-2-e',  # ZCU102 (Zynq UltraScale+ MPSoC)
    backend='Vivado',
    io_type='io_stream',
)

print("\nCompiling hls4ml C-sim (this can take a minute)...")
hls_model.compile()

# hls4ml's channels-last transform means input needs reshaping to (1, 28, 28, 1)
hls_input = x_np.reshape(1, 28, 28, 1)
y_hls = hls_model.predict(hls_input).flatten()

print(f"hls4ml output:          {y_hls}")
print(f"hls4ml argmax:          {np.argmax(y_hls)}")

# --- 4. Compare ---
abs_diff = np.abs(y_ref - y_hls)
print(f"\nMax abs difference:     {abs_diff.max():.6f}")
print(f"Mean abs difference:    {abs_diff.mean():.6f}")
print(f"Argmax match:           {np.argmax(y_ref) == np.argmax(y_hls)}")

if abs_diff.max() < 1e-2:
    print("\nRESULT: Outputs closely match. io_stream produces numerically correct results for cnn_skip.")
else:
    print("\nRESULT: Outputs diverge. The 'Failed to propagate quantization bias down Add node' warning")
    print("is likely the cause — the skip-connection Add is probably not being handled correctly.")