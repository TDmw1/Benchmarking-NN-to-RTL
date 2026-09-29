import os
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat
from dataset import get_eval_dataset

# 1. Define the Calibration Reader to feed data during quantization
class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.dataloader = iter(dataloader)
        self.input_name = input_name

    def get_next(self):
        try:
            images, _ = next(self.dataloader)
            # The quantizer only needs the inputs, not the labels
            return {self.input_name: images.numpy()}
        except StopIteration:
            return None

MODELS = ["mlp_small", "cnn_lenet", "cnn_skip", "cnn_unusual_op"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data_tables", "accuracy_fidelity.csv")
os.makedirs("quantized/qonnx", exist_ok=True)

# 2. Setup dataloaders
calib_loader = get_eval_dataset(batch_size=1, num_samples=100)
eval_loader = get_eval_dataset(batch_size=1, num_samples=1000)

for model_name in MODELS:
    src_onnx = f"models/{model_name}/model.onnx"
    stripped_onnx = f"quantized/qonnx/{model_name}_stripped.onnx"
    dst_onnx = f"quantized/qonnx/{model_name}_int8.onnx"

    model_proto = onnx.load(src_onnx)
    model_proto.graph.ClearField("value_info")
    onnx.save(model_proto, stripped_onnx)

    # Setup the calibrator pointing to the first input node
    session = ort.InferenceSession(stripped_onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Re-initialize the iterator for each model
    calib_reader = ImageCalibrationDataReader(get_eval_dataset(batch_size=1, num_samples=100), input_name)

    # 3. Use quantize_static with QDQ format
    quantize_static(
        model_input=stripped_onnx,
        model_output=dst_onnx,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ, # Explicitly forces Quantize/Dequantize node pairs (hls4ml prefers this)
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )

    if os.path.exists(stripped_onnx):
        os.remove(stripped_onnx)

    # Evaluate quantized model
    q_session = ort.InferenceSession(dst_onnx, providers=["CPUExecutionProvider"])
    q_input_name = q_session.get_inputs()[0].name

    correct, total = 0, 0
    for images, labels in eval_loader:
        out = q_session.run(None, {q_input_name: images.numpy()})[0]
        correct += int(np.argmax(out, axis=1)[0] == labels.numpy()[0])
        total += 1

    q_acc = round((correct / total) * 100.0, 2)
    print(f"[QONNX - {model_name}] Quantized Accuracy: {q_acc:.2f}%")

    # Update CSV table
    df = pd.read_csv(CSV_PATH)
    mask = (df["model_name"] == model_name) & (df["quant_tool"] == "qonnx")
    baseline = df.loc[mask, "fp32_baseline_accuracy"].values[0]
    df.loc[mask, "quantized_accuracy"] = q_acc
    df.loc[mask, "accuracy_delta"] = round(q_acc - baseline, 2)
    df.to_csv(CSV_PATH, index=False)

print("\nPath A (QONNX Static) completed and recorded.")