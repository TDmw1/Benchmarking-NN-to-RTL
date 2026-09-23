# CPU Pipeline Validation

## Summary

The CPU benchmarking pipeline was validated on macOS across all four models
(`mlp_small`, `cnn_lenet`, `cnn_skip`, `cnn_unusual_op`), in both FP32 and
INT8 (QONNX), using ONNX Runtime's CPU Execution Provider. Two things seen 
out of this that were worth looking down into before wrapping up: an odd
init-time number, and a consistent INT8-slower-than-FP32 pattern across all
three CNNs.

---

## Init-time descrepancy — resolved, not a real per-model cost

Early runs showed one model's FP32 init time sitting way out of line with
everything else (20-25ms vs ~1-2ms for the rest). On a repeat run, the same
kind of spike showed up again, but attached to a *different* model —
whichever one happened to run first in that particular script execution.
That's a clear signature of a one-time ONNX Runtime process startup cost
(library load, thread pool init, etc.), not something specific to any one
model. Init time is being treated as a single process-level number from
here on, not a per-model metric.

---

## INT8 running slower than FP32 on CPU

All three CNNs showed INT8 consistently slower than FP32. `mlp_small` did
not. First pass at this used a small number of trials and gave a shaky
signal on `cnn_lenet` specifically (the delta was smaller than the
stddev). Re-ran everything at N=500 per model/precision with proper
median/mean/stddev/SEM to settle it:

| Model | FP32 → INT8 | Result |
|---|---|---|
| `cnn_lenet` | ~3.0x slower | strong indicator, ~8-9 SEMs apart |
| `cnn_skip` | ~2.6-2.75x slower | strong indicator |
| `cnn_unusual_op` | ~6.2x slower | strongest indicator, biggest gap |
| `mlp_small` | ~parity | statistically detectable difference, but the actual gap (~0.2µs) is too small to matter in practice |

`cnn_lenet`'s INT8 numbers also showed a mean noticeably higher than the
median with a heavier right tail than any other row. Not looked into further,
however, potential revisiting may be needed.

---

## Why INT8 is slower — checked with real trace data

Ran ONNX Runtime's built-in profiler on `cnn_skip` (both FP32 and INT8),
then clicked into the actual per-node breakdown in Perfetto for both.

**INT8 trace**: shows explicit `QuantizeLinear`/`DequantizeLinear` nodes at
the model boundary, plus extra sub-costs inside the quantized conv
kernel itself (`quant_scales_mul_kernel_time`,
`output_quantized_cast_kernel_time`) — real, separately-timed work that
has to happen around and inside the actual convolution. Traces can be found
in docs/screenshots where "QSkip..." are trace screenshots of the Quantized model.

**FP32 trace**: same model, same conv/relu/pool/linear sequence, but none
of the above. Clean, direct node sequence, no extra quantize/cast/rescale
steps anywhere.

Side by side, this appears to be a confirmed mechanism,where
the general-purpose CPU doesn't have a true
single-instruction INT8 fast path here, so even a "fused" quantized conv
op still pays extra internal costs, on top of the boundary
quantize/dequantize nodes that don't exist at all in the FP32 graph.

Didn't try to determine the exact percentage of the slowdown these specific
nodes account for — eyeballing a few individual runs put FP32 around
40-45µs and INT8 around 140µs for `cnn_skip`, roughly in line with the
statistical numbers above, which is enough to call this a confirmed
contributing mechanism. Not fully claimed it's the full, precisely-quantified
explanation, as I didn't do the node-level summing that would be needed to say
that with certainty.

---

## Main Point and Caveat

All of this ran on a MacBook, not the actual target edge board. The
pipeline and methodology are solid and validated for moving forward the actual numbers here
are not the final `control_baseline.csv` data. Real board numbers still
need to be collected once hardware is available.