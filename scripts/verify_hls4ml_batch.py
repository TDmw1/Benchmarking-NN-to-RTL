import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

import hls4ml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.core.onnx_exec import execute_onnx

# --- Config ---
INT8_DIR = "quantized/qonnx"
HLS_PROJECT_BASE = "hls_projects/hls4ml_int8_batch_verify"
N_SAMPLES = 100  # bump to 500-1000 once this runs cleanly; C-sim is slow per-sample so start small
NORMALIZE_MEAN = (0.1307,)  # match whatever build_qonnx_models.py used for quick_train()
NORMALIZE_STD = (0.3081,)


def run_model(model_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    onnx_path = os.path.join(INT8_DIR, f"{model_name}_int8.onnx")
    output_dir = os.path.join(HLS_PROJECT_BASE, model_name)

    if not os.path.exists(onnx_path):
        print(f"[hls4ml] Skipping {model_name}: {onnx_path} not found.")
        return

    print(f"\n=== {model_name}: building hls4ml project (io_parallel, default) ===")

    qonnx_model = ModelWrapper(onnx_path)
    qonnx_model = cleanup_model(qonnx_model)
    input_name = qonnx_model.graph.input[0].name
    output_name = qonnx_model.graph.output[0].name

    hls_wrapper = ModelWrapper(onnx_path)
    hls_wrapper = cleanup_model(hls_wrapper)
    hls_wrapper = hls_wrapper.transform(ConvertToChannelsLastAndClean())
    hls_wrapper = hls_wrapper.transform(GemmToMatMul())
    hls_wrapper = cleanup_model(hls_wrapper)

    config = hls4ml.utils.config_from_onnx_model(hls_wrapper, granularity='name', backend='Vivado')
    config['Model']['ReuseFactor'] = 1

    # Widen precision for the fused Add (scale+bias applied post-MatMul) and activation-quant
    # layers: hls4ml's default 'model_default_t' (ap_fixed<16,6>, 10 fractional bits, resolution
    # ~0.001) silently rounds real QONNX quant scales smaller than that (e.g. ~0.0002) to zero,
    # which zeroes out the entire weighted contribution and leaves only the bias term.
    # Confirmed fix on mlp_small: 12.0% -> 100.0% argmax agreement. Apply the same override
    # generically to every Add_N / Quant_N layer present, not just the mlp_small-specific set,
    # so it generalizes to cnn_lenet/cnn_skip's deeper topologies.
    for layer_name in list(config['LayerName'].keys()):
        if (layer_name.startswith('Add_') or layer_name.startswith('Quant_')) and '_param' not in layer_name:
            config['LayerName'][layer_name].setdefault('Precision', {})
            config['LayerName'][layer_name]['Precision']['scale'] = 'ap_fixed<32,16>'
            config['LayerName'][layer_name]['Precision']['bias'] = 'ap_fixed<32,16>'

    convert_kwargs = dict(
        hls_config=config,
        output_dir=output_dir,
        part='xczu9eg-ffvb1156-2-e',  # ZCU102 (Zynq UltraScale+ MPSoC)
        backend='Vivado',
    )
    if model_name == "cnn_skip":
        # io_parallel crashes on cnn_skip's skip-connection Add (unrelated array-split bug);
        # io_stream is the only mode that compiles for this architecture.
        convert_kwargs['io_type'] = 'io_stream'

    hls_model = hls4ml.converters.convert_from_onnx_model(hls_wrapper, **convert_kwargs)
    hls_model.write()
    print(f"Compiling C-sim for {model_name}...")
    hls_model.compile()

    correct = 0
    max_abs_diffs = []
    mean_abs_diffs = []
    argmax_mismatches = []

    for i in range(N_SAMPLES):
        x_sample, y_label = test_set[i]
        x_np = x_sample.numpy().astype(np.float32)

        # QONNX reference
        ref_input = x_np.reshape(1, 1, 28, 28)
        ref_out = execute_onnx(qonnx_model, {input_name: ref_input})[output_name].flatten()

        # hls4ml C-sim - input layout depends on whether the model is 2D (MLP, no channels-last transform
        # needed) or 4D (CNN, channels-last after ConvertToChannelsLastAndClean)
        if x_np.size == 28 * 28 and "mlp" in model_name:
            hls_input = x_np.reshape(1, 28 * 28)
        else:
            hls_input = x_np.reshape(1, 28, 28, 1)

        hls_out = hls_model.predict(hls_input).flatten()

        abs_diff = np.abs(ref_out - hls_out)
        max_abs_diffs.append(abs_diff.max())
        mean_abs_diffs.append(abs_diff.mean())

        ref_pred = np.argmax(ref_out)
        hls_pred = np.argmax(hls_out)
        if ref_pred == hls_pred:
            correct += 1
        else:
            argmax_mismatches.append((i, int(y_label), int(ref_pred), int(hls_pred)))

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{N_SAMPLES} samples done")

    agreement_rate = correct / N_SAMPLES * 100
    print(f"\n--- {model_name} results over {N_SAMPLES} samples ---")
    print(f"hls4ml vs QONNX argmax agreement: {agreement_rate:.1f}%")
    print(f"Mean of per-sample max abs diff:  {np.mean(max_abs_diffs):.6f}")
    print(f"Mean of per-sample mean abs diff: {np.mean(mean_abs_diffs):.6f}")
    if argmax_mismatches:
        print(f"First few mismatches (idx, true_label, qonnx_pred, hls4ml_pred): {argmax_mismatches[:5]}")

    return {
        "model_name": model_name,
        "n_samples": N_SAMPLES,
        "argmax_agreement_pct": agreement_rate,
        "mean_max_abs_diff": float(np.mean(max_abs_diffs)),
        "mean_mean_abs_diff": float(np.mean(mean_abs_diffs)),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("mlp_small", "cnn_lenet", "cnn_skip"):
        print("Usage: python scripts/verify_hls4ml_batch.py <mlp_small|cnn_lenet|cnn_skip>")
        print("Run each model as its own separate process invocation:")
        print("  python scripts/verify_hls4ml_batch.py mlp_small")
        print("  python scripts/verify_hls4ml_batch.py cnn_lenet")
        print("  python scripts/verify_hls4ml_batch.py cnn_skip")
        sys.exit(1)

    model_name = sys.argv[1]
    result = run_model(model_name)
    print("\n\n=== Result ===")
    print(result)