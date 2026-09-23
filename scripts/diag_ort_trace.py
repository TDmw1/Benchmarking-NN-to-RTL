import onnxruntime as ort
import numpy as np

model_path = "quantized/qonnx/cnn_skip_int8.onnx"

# Enable ONNX Runtime Profiling
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

print("Starting ORT Profiling run for cnn_skip_int8...")
sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

input_meta = sess.get_inputs()[0]
# Provide the float32 input the model expects at its boundary
dummy_input = {input_meta.name: np.random.randn(1, 1, 28, 28).astype(np.float32)}

for _ in range(10000):
    sess.run(None, dummy_input)

prof_file = sess.end_profiling()
print(f"Trace saved to: {prof_file}")