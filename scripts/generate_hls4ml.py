import os
import hls4ml
import qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean

MODELS = ["mlp_small", "cnn_lenet", "cnn_skip", "cnn_unusual_op"]
INT8_DIR = "quantized/qonnx"
OUTPUT_BASE = "hls_projects/hls4ml_int8"

os.makedirs(OUTPUT_BASE, exist_ok=True)

for model_name in MODELS:
    onnx_path = os.path.join(INT8_DIR, f"{model_name}_int8.onnx")
    output_dir = os.path.join(OUTPUT_BASE, model_name)
    
    if not os.path.exists(onnx_path):
        print(f"[hls4ml] Skipping {model_name}: {onnx_path} not found.")
        continue

    print(f"\n--- hls4ml: Converting {model_name} (Static INT8) ---")
    try:
        # Load the quantized model
        model = ModelWrapper(onnx_path)
        
        # 1. Clean the graph (infers shapes, folds constants)
        model = cleanup_model(model)
        
        # 2. Apply explicit hardware formatting transformations
        model = model.transform(ConvertToChannelsLastAndClean())
        model = model.transform(GemmToMatMul())
        
        # 3. Clean the graph one more time after transformations
        model = cleanup_model(model)
        
        # 4. Parse ONNX graph to hls4ml config
        # CRITICAL CHANGE: default_precision removed so hls4ml uses the baked-in INT8 scales
        config = hls4ml.utils.config_from_onnx_model(
            model, 
            granularity='name',
            backend='Vivado'
        )
        config['Model']['ReuseFactor'] = 1
        config['Model']['Strategy'] = 'Resource'  # sometimes changes which conv codegen path is used
        
        # 5. Setup converter
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model,
            hls_config=config,
            output_dir=output_dir,
            part='xczu9eg-ffvb1156-2-e',  # ZCU102 (Zynq UltraScale+ MPSoC)
            backend='Vivado'
        )
        
        # 6. Write C++ project files to disk ONLY
        hls_model.write()
        print(f"✅ Success: HLS C++ project written to {output_dir}/")
        
    except Exception as e:
        import traceback
        print(f"❌ Failed: {model_name} threw an error: {e}")
        traceback.print_exc()