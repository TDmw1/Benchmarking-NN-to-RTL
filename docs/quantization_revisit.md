# Architecture Revisit: The Move to Static QDQ

## Summary

The first pass at Phase 5 quantization worked fine for the CPU (using `quantize_dynamic`), but hit a wall when feeding those graphs into `hls4ml`. Dynamic quantization calculates scale factors on the fly. FPGAs need those scale factors hardcoded at compile time to map operations to DSP slices effectively. 

To fix this, the Phase 5 export script (`quantize_qonnx.py`) was rewritten to use ONNX Runtime's `quantize_static` with a calibration reader. This uses the scaling constants directly into the graph using the QDQ (Quantize/Dequantize) format, which is the structure the FPGA toolchains expect.

---

## Opset 27 Crash and the Process Initialization Barrier

Re-running the CPU validation audit (`audit_cpu_validation.py`) on the new static models immediately crashed during the `burn_process_init_overhead()` function.

The issue was a package version mismatch. The newer `onnx` Python package defaults to Opset 27, but our `onnxruntime` environment only guarantees stability up to Opset 26. Since the script dynamically builds a tiny dummy graph to measure OS thread-pool startup time, it didn't have an opset version stamped on it, causing the crash.

**The Fix:** Patched `audit_cpu_validation.py` to explicitly force `opset 18` when generating the dummy model. This cleared the error, isolated the ~2ms startup barrier, and let the sub-microsecond audits finish.

---

## Persistent Trend: INT8 models run slower than FP32

The old `profile_cpu_validation.py` data showed dynamic INT8 running way slower than FP32 on the Mac CPU. The new `audit_cpu_validation.py` data (N=500, sub-microsecond precision) shows that **Static QDQ INT8 is also consistently slower than FP32** for the CNNs.

| Model | FP32 Mean Latency | Static INT8 (QDQ) | Result |
|---|---|---|---|
| `mlp_small` | 7.01 µs | 7.86 µs | ~Parity |
| `cnn_lenet` | 19.72 µs | 43.08 µs | ~2.2x slower |
| `cnn_skip` | 22.11 µs | 44.84 µs | ~2.0x slower |
| `cnn_unusual_op` | 25.02 µs | 101.06 µs | ~4.0x slower |

**Why this is happening:**
With dynamic quantization, the CPU was most likely bogged down when calculating min/max ranges on the fly. With static QDQ, the ranges are fixed, but the graph now has explicit `QuantizeLinear` and `DequantizeLinear` nodes surrounding every convolution. The CPU has to constantly cast float tensors to int8, do the math, and cast back to float. This packing/unpacking overhead is dominating the execution time.

---

## Looking Forward

This latency issue is a big point why the hardware benchmark exists. The static QDQ format that bogs down the CPU with casting operations is what the FPGA compilers (`hls4ml`, `nn2FPGA`) need to see. They strip those casting nodes out and map the static scales as hardwired bit-shifts directly against the DSP slices.
