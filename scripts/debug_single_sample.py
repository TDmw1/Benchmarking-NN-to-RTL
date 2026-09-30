import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.core.onnx_exec import execute_onnx
import hls4ml

onnx_path = "quantized/qonnx/mlp_small_int8.onnx"
qonnx_model = cleanup_model(ModelWrapper(onnx_path))
input_name = qonnx_model.graph.input[0].name
output_name = qonnx_model.graph.output[0].name

hls_wrapper = cleanup_model(ModelWrapper(onnx_path))
hls_wrapper = hls_wrapper.transform(ConvertToChannelsLastAndClean())
hls_wrapper = hls_wrapper.transform(GemmToMatMul())
hls_wrapper = cleanup_model(hls_wrapper)

config = hls4ml.utils.config_from_onnx_model(hls_wrapper, granularity='name', backend='Vivado')
config['Model']['ReuseFactor'] = 1
hls_model = hls4ml.converters.convert_from_onnx_model(
    hls_wrapper, hls_config=config,
    output_dir="hls_projects/hls4ml_int8_batch_verify/mlp_small",
    part='xczu9eg-ffvb1156-2-e', backend='Vivado',
)
hls_model.compile()

import torch
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
x_sample, y_label = test_set[0]
x_np = x_sample.numpy().astype(np.float32)

ref_out = execute_onnx(qonnx_model, {input_name: x_np.reshape(1,1,28,28)})[output_name].flatten()
hls_out = hls_model.predict(x_np.reshape(1, 28*28)).flatten()

print("QONNX reference:", ref_out)
print("hls4ml output:  ", hls_out)
print("ratio (hls/ref):", hls_out / ref_out)