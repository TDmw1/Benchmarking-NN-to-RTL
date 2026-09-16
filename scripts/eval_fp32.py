import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from dataset import get_eval_dataset

MODELS = ["mlp_small", "cnn_lenet", "cnn_skip", "cnn_unusual_op"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data_tables", "accuracy_fidelity.csv")

def evaluate_onnx_model(onnx_path, loader):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    correct = 0
    total = 0
    
    for images, labels in loader:
        # Convert PyTorch tensor to NumPy array matching ONNX input
        ort_inputs = {input_name: images.numpy()}
        ort_outs = session.run(None, ort_inputs)
        
        preds = np.argmax(ort_outs[0], axis=1)
        correct += np.sum(preds == labels.numpy())
        total += labels.size(0)
        
    accuracy = (correct / total) * 100.0
    return accuracy

def main():
    loader = get_eval_dataset(batch_size=1, num_samples=1000)
    accuracies = {}
    
    for model_name in MODELS:
        onnx_path = os.path.join("models", model_name, "model.onnx")
        acc = evaluate_onnx_model(onnx_path, loader)
        accuracies[model_name] = acc
        print(f"[{model_name}] FP32 Accuracy (1000 samples): {acc:.2f}%")
        
    # Update accuracy_fidelity.csv
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        for model_name, acc in accuracies.items():
            df.loc[df["model_name"] == model_name, "fp32_baseline_accuracy"] = round(acc, 2)
        df.to_csv(CSV_PATH, index=False)
        print(f"\nUpdated {CSV_PATH} with FP32 baseline values.")
    else:
        print(f"Warning: {CSV_PATH} not found. Please create the CSV skeleton first.")

if __name__ == "__main__":
    main()