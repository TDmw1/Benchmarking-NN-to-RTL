import os
import numpy as np
import pandas as pd
import onnx
from onnxsim import simplify
import nngen as ng
from dataset import get_eval_dataset

MODELS = ["mlp_small", "cnn_lenet", "cnn_skip", "cnn_unusual_op"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data_tables", "accuracy_fidelity.csv")
os.makedirs("quantized/nngen", exist_ok=True)

# Standard MNIST normalization statistics
# Change to 0.0 and 1.0 if you didn't use transforms.Normalize in training
DATASET_MEAN = np.array([0.1307]).astype(np.float32)
DATASET_STD = np.array([0.3081]).astype(np.float32)

loader = get_eval_dataset(batch_size=1, num_samples=1000)

for model_name in MODELS:
    onnx_path = f"models/{model_name}/model.onnx"
    sim_onnx_path = f"quantized/nngen/{model_name}_sim.onnx"

    q_acc = None
    try:
        model_proto = onnx.load(onnx_path)
        model_simp, check = simplify(model_proto)
        onnx.save(model_simp, sim_onnx_path)

        outputs, placeholders, variables, constants, operators = ng.from_onnx(
            sim_onnx_path, value_dtypes={"input": ng.int8}
        )

        input_node = placeholders["input"]
        input_name = input_node.name

        print(f"[{model_name}] real output candidates:")
        for key, node in outputs.items():
            print(f"    key='{key}'  shape={node.shape}  dtype={node.dtype}")

        # With the correct dict, there should be exactly one entry
        # corresponding to the ONNX graph's declared output tensor.
        output_key, output_node = list(outputs.items())[0]
        print(f"[{model_name}] selected output key='{output_key}' shape={output_node.shape}")

        act_scale_factor = int(round(2 ** (ng.int8.width - 1) * 0.5))  # 64 for int8
        input_scale_factors = {input_name: act_scale_factor}
        input_means = {input_name: DATASET_MEAN * act_scale_factor}
        input_stds = {input_name: DATASET_STD * act_scale_factor}

        ng.quantize([output_node], input_scale_factors, input_means, input_stds)
        print(f"[{model_name}] output_node dtype after quantize: {output_node.dtype}")

        # --- Sanity check on ONE sample before running the full loop ---
        sample_img, sample_label = next(iter(loader))
        sample_int = np.clip(
            np.round(sample_img.numpy() * act_scale_factor), -128, 127
        ).astype(np.int64)
        sanity_out = ng.eval([output_node], **{input_name: sample_int})
        print(f"[{model_name}] sanity check raw output (shape {sanity_out[0].shape}): "
              f"{sanity_out[0].flatten()}")
        # Expect NUM_CLASSES=10 integer-valued entries with real variation
        # that actually depend on the input image this time.

        correct, total = 0, 0
        for images, labels in loader:
            input_data_int = np.clip(
                np.round(images.numpy() * act_scale_factor), -128, 127
            ).astype(np.int64)

            eval_outs = ng.eval([output_node], **{input_name: input_data_int})
            pred = np.argmax(eval_outs[0])
            correct += int(pred == labels.numpy()[0])
            total += 1

        q_acc = round((correct / total) * 100.0, 2)
        print(f"[NNgen Pow2 - {model_name}] Quantized Accuracy: {q_acc:.2f}%")

    except Exception as e:
        import traceback
        print(f"[NNgen Pow2 - {model_name}] Failed: {e}")
        traceback.print_exc()

    if os.path.exists(sim_onnx_path):
        os.remove(sim_onnx_path)

    df = pd.read_csv(CSV_PATH)
    mask = (df["model_name"] == model_name) & (df["quant_tool"] == "nngen")
    baseline = df.loc[mask, "fp32_baseline_accuracy"].values[0]

    if q_acc is not None:
        df.loc[mask, "quantized_accuracy"] = q_acc
        df.loc[mask, "accuracy_delta"] = round(q_acc - baseline, 2)
    else:
        df.loc[mask, "quantized_accuracy"] = np.nan
        df.loc[mask, "accuracy_delta"] = np.nan
    df.to_csv(CSV_PATH, index=False)

print("\nPath B (NNgen) completed and recorded.")