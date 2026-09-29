import os
import subprocess

MODELS = ["mlp_small", "cnn_lenet", "cnn_skip", "cnn_unusual_op"]
FP32_DIR = "models"
OUTPUT_BASE = "nn2fpga_projects/nn2fpga_fp32_baseline"

os.makedirs(OUTPUT_BASE, exist_ok=True)

for model_name in MODELS:
    onnx_path = os.path.abspath(os.path.join(FP32_DIR, model_name, "model.onnx"))
    project_root = os.path.abspath(os.path.join(OUTPUT_BASE, model_name))
    config_path = os.path.abspath(f"temp_config_{model_name}.toml")
    
    if not os.path.exists(onnx_path):
        print(f"[nn2FPGA] Skipping {model_name}: {onnx_path} not found.")
        continue

    toml_content = f"""[project]
top_name = "{model_name}"
onnx_path = "{onnx_path}"
project_root = "{project_root}"
hls_version = 2025.2

[platform]
board = "KRIA"
frequency = 250

[steps]
OptimizeBitwidth = true
AddStreamingParams = true
ComputeFifoDepth = true
OptimizeFifo = true
Simulate = false
GenerateBitstream = false
Deploy = false

[options]
silvia_packing = true
"""
    
    with open(config_path, "w") as f:
        f.write(toml_content)

    print(f"\n--- nn2FPGA: Parsing {model_name} (Plain FP32) ---")
    try:
        # This is the line that was broken and is now fixed:
        cmd = [
            "python", "-m", "nn2fpga.compiler.cli",
            "--config", config_path
        ]
        
        env = os.environ.copy()
        env["XILINX_VERSION"] = "2025.2"
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success: nn2FPGA successfully parsed and compiled {model_name}!")
        else:
            print(f"❌ Failed: {model_name} parser threw an error:")
            print("\n".join(result.stderr.strip().split("\n")[-15:]))
            
    except Exception as e:
        print(f"❌ Execution Failed: {e}")
    
    if os.path.exists(config_path):
        os.remove(config_path)