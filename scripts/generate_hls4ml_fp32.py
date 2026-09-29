import os
import hls4ml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul

MODELS = ["cnn_lenet", "cnn_skip", "cnn_unusual_op", "mlp_small"]
FP32_DIR = "models" 
OUTPUT_BASE = "hls4ml_projects/hls4ml_fp32_baseline"

os.makedirs(OUTPUT_BASE, exist_ok=True)

for model_name in MODELS:
    # Target the specific nested structure from the image
    raw_path = os.path.join(FP32_DIR, model_name, "model.onnx")
    output_dir = os.path.join(OUTPUT_BASE, model_name)
    
    if not os.path.exists(raw_path):
        print(f"Skipping {model_name}: {raw_path} not found.")
        continue

    print(f"\n--- Processing Plain FP32 Model: {model_name} ---")
    try:
        model = ModelWrapper(raw_path)
        model = cleanup_model(model)
        
        # Hardware-Specific Topographic Transformations
        model = model.transform(ConvertToChannelsLastAndClean())
        model = model.transform(GemmToMatMul())
        model = cleanup_model(model)
        
        # Generate the hls4ml configuration (automatically quantizes to ap_fixed<16,6>)
        config = hls4ml.utils.config_from_onnx_model(model, granularity='name', backend='Vivado')
        
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model,
            hls_config=config,
            output_dir=output_dir,
            part='xcvu9p-flgb2104-2-e',
            backend='Vivado'
        )
        
        hls_model.write()
        print(f"✅ Success: HLS C++ project written to {output_dir}/")
        
    except Exception as e:
        print(f"❌ Failed to convert {model_name}: {e}")