import os
import traceback
import hls4ml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

MODELS = ["cnn_lenet", "cnn_skip", "cnn_unusual_op", "mlp_small"]
INT8_DIR = "quantized/qonnx"
OUTPUT_BASE = "hls_projects/hls4ml_int8_fixed"

os.makedirs(OUTPUT_BASE, exist_ok=True)

for model_name in MODELS:
    raw_path = os.path.join(INT8_DIR, f"{model_name}_int8.onnx")
    output_dir = os.path.join(OUTPUT_BASE, model_name)
    
    if not os.path.exists(raw_path):
        print(f"Skipping {model_name}: {raw_path} not found.")
        continue

    print(f"\n--- Pre-processing & Converting {model_name} ---")
    try:
        model = ModelWrapper(raw_path)
        
        # 1. Infer shapes and types BEFORE the QCDQ transform
        model = cleanup_model(model)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        
        # 2. Transform standard ONNX QDQ into hls4ml QONNX Quant nodes
        model = model.transform(QCDQToQuant())
        
        # 3. Clean up any loose nodes left by the transformation
        model = cleanup_model(model)
        
        # 4. Hardware-Specific Topographic Transformations
        model = model.transform(ConvertToChannelsLastAndClean())
        model = model.transform(GemmToMatMul())
        model = cleanup_model(model)
        
        # 5. Generate the hls4ml C++ project
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
        traceback.print_exc()