import os
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from dataset import get_eval_dataset

MODELS = ["mlp_small", "cnn_lenet", "cnn_skip", "cnn_unusual_op"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data_tables", "accuracy_fidelity.csv")
os.makedirs("quantized/qonnx", exist_ok=True)

loader = get_eval_dataset(batch_size=1, num_samples=1000)

for model_name in MODELS:
    src_onnx = f"models/{model_name}/model.onnx"
    stripped_onnx = f"quantized/qonnx/{model_name}_stripped.onnx"
    dst_onnx = f"quantized/qonnx/{model_name}_int8.onnx"

    # Manually strip conflicting shape metadata from the ONNX graph
    model_proto = onnx.load(src_onnx)
    model_proto.graph.ClearField("value_info")
    onnx.save(model_proto, stripped_onnx)

    # Quantize the clean model
    quantize_dynamic(
        model_input=stripped_onnx,
        model_output=dst_onnx,
        weight_type=QuantType.QInt8
    )

    # Clean up the intermediate file
    if os.path.exists(stripped_onnx):
        os.remove(stripped_onnx)

    #  Evaluate quantized model
    session = ort.InferenceSession(dst_onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    correct, total = 0, 0
    for images, labels in loader:
        out = session.run(None, {input_name: images.numpy()})[0]
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

print("\nPath A (QONNX) completed and recorded.")