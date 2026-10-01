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

# A single global Model-level ReuseFactor doesn't work across layers of very different sizes:
# it must be a valid divisor of each layer's own reduction dimension for the Resource-strategy
# codegen to actually share multipliers, otherwise hls4ml silently falls back to fully unrolling
# that layer. Confirmed this session: ReuseFactor=64 at the Model level still fully unrolled
# MatMul_0 (784x64) down to the input dimension, producing a ~3 million instruction design that
# would never finish scheduling. Fix: set ReuseFactor PER LAYER to that layer's own input size
# (n_in) -- always a valid value -- and force Strategy='Resource' so multipliers are reused
# serially over n_in cycles instead of laid out in parallel. This caps every Dense/MatMul layer
# at n_out multipliers, however large its input is.
DEFAULT_MAX_REUSE_FACTOR = None  # None = use full n_in per layer (fewest DSPs); cap it via CLI arg

# NOTE: backend='Vitis', not 'Vivado' — Vivado 2026.1 ships Vitis HLS (vitis_hls), not the
# deprecated Vivado HLS (vivado_hls). Confirm `vitis-run --version` works in this shell first.

# NOTE on part: xczu9eg-ffvb1156-2-e (ZCU102) requires a CORE-tier (or higher) Vivado license.
# Our local license is BASIC (free) tier, which only covers Zynq UltraScale+ MPSoC parts ZU1-ZU7.
# Using xczu3eg-sbva484-1-e (Ultra96-V2, ZU3EG) here instead — same Zynq UltraScale+ MPSoC family
# as the ZCU102's ZU9EG, so the toolchain/codegen path is representative, just a smaller/cheaper
# member of the family that's actually licensed to synthesize on this machine right now. Swap back
# to xczu9eg-ffvb1156-2-e once a CORE-tier (or department/university) license is in place.
DEFAULT_PART = "xczu3eg-sbva484-1-e"  # Ultra96-V2 (Zynq UltraScale+ ZU3EG) - BASIC-tier licensed


def apply_per_layer_resource_reuse(hls_wrapper, config, max_reuse_factor=None):
    """Force Strategy='Resource' and set each MatMul layer's ReuseFactor to its own n_in
    (optionally capped at max_reuse_factor), so every Dense-equivalent layer reuses a small,
    valid pool of multipliers instead of hls4ml silently fully-unrolling an incompatible layer.
    """
    config['Model']['Strategy'] = 'Resource'
    for node in hls_wrapper.graph.node:
        if node.op_type != 'MatMul':
            continue
        layer_name = node.name
        if layer_name not in config['LayerName']:
            continue
        weight_name = node.input[1]
        try:
            n_in, n_out = hls_wrapper.get_tensor_shape(weight_name)
        except Exception as exc:
            print(f"[reuse] Skipping {layer_name}: couldn't read weight shape ({exc})")
            continue
        rf = n_in if max_reuse_factor is None else min(n_in, max_reuse_factor)
        config['LayerName'][layer_name]['Strategy'] = 'Resource'
        config['LayerName'][layer_name]['ReuseFactor'] = rf
        print(f"[reuse] {layer_name}: n_in={n_in}, n_out={n_out} -> ReuseFactor={rf} (Resource strategy, "
              f"~{n_out} multiplier(s) reused over {rf} cycle(s))")


def build_model(model_name, part=DEFAULT_PART, max_reuse_factor=DEFAULT_MAX_REUSE_FACTOR):
    onnx_path = os.path.join(INT8_DIR, f"{model_name}_int8.onnx")
    output_dir = os.path.join(HLS_PROJECT_BASE, model_name)

    if not os.path.exists(onnx_path):
        print(f"[hls4ml] Skipping {model_name}: {onnx_path} not found.")
        return

    print(f"\n=== {model_name}: building + synthesizing (backend=Vitis, part={part}, "
          f"max_reuse_factor={max_reuse_factor}) ===")

    hls_wrapper = ModelWrapper(onnx_path)
    hls_wrapper = cleanup_model(hls_wrapper)
    hls_wrapper = hls_wrapper.transform(ConvertToChannelsLastAndClean())
    hls_wrapper = hls_wrapper.transform(GemmToMatMul())
    hls_wrapper = cleanup_model(hls_wrapper)

    config = hls4ml.utils.config_from_onnx_model(hls_wrapper, granularity='name', backend='Vitis')
    apply_per_layer_resource_reuse(hls_wrapper, config, max_reuse_factor=max_reuse_factor)

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
        part=part,
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
    if len(sys.argv) < 2 or sys.argv[1] not in ("mlp_small", "cnn_lenet"):
        print("Usage: python scripts/generate_hls4ml_synthesis.py <mlp_small|cnn_lenet> [part] [max_reuse_factor]")
        print(f"  If [part] is omitted, defaults to {DEFAULT_PART} (BASIC-tier licensed).")
        print("  Each MatMul layer's ReuseFactor is auto-set to its own input size (n_in) by default --")
        print("  the fewest DSPs, most serial. Pass [max_reuse_factor] to cap it lower for more parallelism.")
        print("Run each model as its own separate process invocation:")
        print("  python scripts/generate_hls4ml_synthesis.py mlp_small")
        print("  python scripts/generate_hls4ml_synthesis.py cnn_lenet")
        print("  python scripts/generate_hls4ml_synthesis.py mlp_small xczu9eg-ffvb1156-2-e   # once CORE-licensed")
        print("  python scripts/generate_hls4ml_synthesis.py mlp_small xczu3eg-sbva484-1-e 64  # cap reuse at 64")
        sys.exit(1)

    part_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PART
    max_reuse_arg = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MAX_REUSE_FACTOR
    build_model(sys.argv[1], part=part_arg, max_reuse_factor=max_reuse_arg)