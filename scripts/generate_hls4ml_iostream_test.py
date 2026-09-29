import os
import hls4ml
import qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean

# --- VERIFY FIRST: confirm convert_from_onnx_model actually accepts io_type ---
# Run this one line alone before trusting the rest of the script:
#   help(hls4ml.converters.convert_from_onnx_model)
# Look for an `io_type` parameter in the signature. If it's not there,
# paste the actual signature back before running the rest of this.

MODELS = ["cnn_skip"]  # scoped to just the one model under test
INT8_DIR = "quantized/qonnx"
OUTPUT_BASE = "hls_projects/hls4ml_int8_iostream_test"

os.makedirs(OUTPUT_BASE, exist_ok=True)

for model_name in MODELS:
    onnx_path = os.path.join(INT8_DIR, f"{model_name}_int8.onnx")
    output_dir = os.path.join(OUTPUT_BASE, model_name)

    if not os.path.exists(onnx_path):
        print(f"[hls4ml] Skipping {model_name}: {onnx_path} not found.")
        continue

    print(f"\n--- hls4ml: Converting {model_name} (Static INT8, io_stream test) ---")
    try:
        model = ModelWrapper(onnx_path)
        model = cleanup_model(model)
        model = model.transform(ConvertToChannelsLastAndClean())
        model = model.transform(GemmToMatMul())
        model = cleanup_model(model)

        config = hls4ml.utils.config_from_onnx_model(
            model,
            granularity='name',
            backend='Vivado'
        )
        config['Model']['ReuseFactor'] = 1  # left in place; already confirmed to have no effect, but harmless

        hls_model = hls4ml.converters.convert_from_onnx_model(
            model,
            hls_config=config,
            output_dir=output_dir,
            part='xcvu9p-flgb2104-2-e',
            backend='Vivado',
            io_type='io_stream',  # <-- the actual change under test this run
        )

        hls_model.write()
        print(f"SUCCESS: HLS C++ project written to {output_dir}/")

    except Exception as e:
        import traceback
        print(f"FAILED: {model_name} threw an error: {e}")
        traceback.print_exc()