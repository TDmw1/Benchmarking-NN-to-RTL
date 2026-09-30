import os
import sys

import hls4ml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean

# --- Config ---
INT8_DIR = "quantized/qonnx"
HLS_PROJECT_BASE = "hls_projects/hls4ml_int8_synthesis"
CLOCK_PERIOD_NS = 10  # 100 MHz; conservative starting point, revisit if timing closes easily

# NOTE: backend='Vitis', not 'Vivado' — Vivado 2026.1 ships Vitis HLS (vitis_hls), not the
# deprecated Vivado HLS (vivado_hls). Confirm `vitis_hls -version` works in this shell first
# (source Xilinx's settings64.bat if it doesn't).


def build_model(model_name):
    onnx_path = os.path.join(INT8_DIR, f"{model_name}_int8.onnx")
    output_dir = os.path.join(HLS_PROJECT_BASE, model_name)

    if not os.path.exists(onnx_path):
        print(f"[hls4ml] Skipping {model_name}: {onnx_path} not found.")
        return

    print(f"\n=== {model_name}: building + synthesizing (backend=Vitis, part=ZCU102) ===")

    hls_wrapper = ModelWrapper(onnx_path)
    hls_wrapper = cleanup_model(hls_wrapper)
    hls_wrapper = hls_wrapper.transform(ConvertToChannelsLastAndClean())
    hls_wrapper = hls_wrapper.transform(GemmToMatMul())
    hls_wrapper = cleanup_model(hls_wrapper)

    config = hls4ml.utils.config_from_onnx_model(hls_wrapper, granularity='name', backend='Vitis')
    config['Model']['ReuseFactor'] = 1

    # Same precision fix confirmed this session: mlp_small 12.0% -> 100.0%, cnn_lenet
    # 11.0% -> 100.0% argmax agreement. model_default_t (ap_fixed<16,6>, 10 fractional bits)
    # silently rounds real QONNX quant scales (~1e-4) to zero in fused Add/ApplyAlpha layers.
    for layer_name in list(config['LayerName'].keys()):
        if (layer_name.startswith('Add_') or layer_name.startswith('Quant_')) and '_param' not in layer_name:
            config['LayerName'][layer_name].setdefault('Precision', {})
            config['LayerName'][layer_name]['Precision']['scale'] = 'ap_fixed<32,16>'
            config['LayerName'][layer_name]['Precision']['bias'] = 'ap_fixed<32,16>'

    hls_model = hls4ml.converters.convert_from_onnx_model(
        hls_wrapper,
        hls_config=config,
        output_dir=output_dir,
        part='xczu9eg-ffvb1156-2-e',  # ZCU102 (Zynq UltraScale+ MPSoC)
        backend='Vitis',
        clock_period=CLOCK_PERIOD_NS,
    )
    hls_model.write()

    print(f"Running build (csim + synth + export) for {model_name}... this can take several minutes.")
    hls_model.build(csim=True, synth=True, cosim=False, export=True)

    report_dir = os.path.join(output_dir, 'myproject_prj', 'solution1', 'syn', 'report')
    print(f"\nLook for the synthesis report under: {report_dir}")
    if os.path.isdir(report_dir):
        for f in os.listdir(report_dir):
            print(f"  - {f}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("mlp_small", "cnn_lenet"):
        print("Usage: python scripts/generate_hls4ml_synthesis.py <mlp_small|cnn_lenet>")
        print("Run each model as its own separate process invocation:")
        print("  python scripts/generate_hls4ml_synthesis.py mlp_small")
        print("  python scripts/generate_hls4ml_synthesis.py cnn_lenet")
        sys.exit(1)

    build_model(sys.argv[1])