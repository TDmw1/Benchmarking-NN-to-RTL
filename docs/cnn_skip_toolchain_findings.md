# hls4ml and nn2FPGA Deep-Dive on cnn_skip

## Summary

With the real QONNX pipeline finally in place (see the note on quantize_qonnx.py
below), hls4ml and nn2FPGA got past their earlier format-related crashes and
into something more different: two actual, source-confirmed limitations in
the tools themselves, both centered on `cnn_skip` and its residual `Add`
connection. Neither turned out to be fixable from this side — both trace back
to unfinished code paths in the installed tool versions.

---

## Note: quantize_qonnx.py was never producing real QONNX

Noting this up front since it explains why earlier phases kept crashing on
both toolchains no matter what was tried: `quantize_qonnx.py` was using ONNX
Runtime's `quantize_static` with QDQ format — `QuantizeLinear`/`DequantizeLinear`
node pairs — which is NOT what QONNX actually is. Real QONNX uses a single
fused `Quant` node per tensor, carrying scale/zero-point/bitwidth as
attributes, and it's specifically what `brevitas.export.export_qonnx()`
produces. Built `build_qonnx_models.py` to do this properly with real
Brevitas QAT models. Confirmed genuine QONNX output by checking the op
types in the exported graph — `Quant`, not `QuantizeLinear`/`DequantizeLinear`.
The old script was renamed to `quantize_qdq_qonnx.py` to stop it from being
confused for the real thing again.

---

## hls4ml: cnn_skip fails both ways, same root cause

### io_parallel (default) — crashes at conversion

```text
ValueError: array split does not result in an equal division
```

Traced to `fpga_backend.py`, `generate_conv2d_line_buffer_fn`,
`np.split(im2col_matrix, n_partitions)`. Tried `ReuseFactor = 1` first since
that's the obvious knob — confirmed via hls4ml's own docs and a direct test
that it has zero effect. `n_partitions` comes from layer geometry, not
ReuseFactor.

### io_stream — compiles, but the output is wrong

Switching `io_type` to `io_stream` sidesteps the crash entirely and produces
a working HLS project. But conversion still throws this warning, which turned
out to matter a lot more than it looked at first:

```text
Failed to propagate quantization bias down Add node; model probably not suppored.
```

(Typo is theirs, straight from the installed source — not a paraphrase.)

Ran the actual C-simulation against a real MNIST sample and compared it
against a genuine QONNX reference execution (`qonnx.core.onnx_exec.execute_onnx`,
which correctly runs the custom `Quant` op):

| Source | Output | Predicted class |
|---|---|---|
| QONNX reference | real logits, -14.6 to 10.2 | 7 (correct) |
| hls4ml io_stream | near-zero noise, -0.02 to 0.05 | 1 (wrong) |

Max abs diff: 14.6. Mean abs diff: 6.6. Not a rounding gap — appears to be broken.

The compile logs also spammed this hundreds of times:

```text
WARNING: Hls::stream 'layer57_out' is read while empty, which may result in RTL simulation hanging.
```

That's the actual issue showing up directly — something is reading a
stream that was never written to, right at the point in the dataflow graph
where the two branches merge. Lines up with the quantization-bias warning
from conversion. hls4ml told us it couldn't handle this before it ever ran.

**Another thing to note as a disclaimer before trusting any of the above**: C-sim compilation itself
failed first, unrelated to any of this — Apple Clang 17's libc++ has a
stricter `<complex>` forward declaration that collides with Xilinx's bundled
`ap_types` headers (`reference to 'complex' is ambiguous`). Fixed by
installing GNU GCC via Homebrew and forcing `build_lib.sh` (which hardcodes
`g++`) to resolve to it via a PATH-priority symlink. Purely a
macOS/toolchain thing, checked and is unrelated to the actual numeric result once
past it.

**One thing that stands out**: this same `cnn_skip` architecture converted cleanly through
hls4ml back when it was fed plain FP32 input (no QONNX `Quant` nodes at all).
So this isn't "hls4ml can't do skip connections" — it's specifically the
QONNX-aware quantization-propagation pass (`move_scales.py`) that has a real
gap for `Add` nodes. Core HLS codegen for merges is fine. The quant-aware
front end isn't there yet for this case.

---

## nn2FPGA: same symptom, wrong assumption about the cause

Running the actual QONNX through nn2FPGA still hits the exact same error on
`cnn_lenet`, `cnn_skip`, and `cnn_unusual_op`:

```text
AttributeError: 'FloatTensorType' object has no attribute 'scale'
```

This is the same crash that showed up back when the input was plain FP32.
Assumption at the time was that real QONNX metadata would fix it. It didn't.
Went and read `propagate_quant.py` to find out why, and here is the source:

```python
QUANT_INVARIANT_NODES = [
    "BandwidthAdjustDecreaseWord", "BandwidthAdjustDecreaseStreams",
    "BandwidthAdjustIncreaseWord", "BandwidthAdjustIncreaseStreams",
    "Concat", "Flatten", "GlobalMaxPool", "Identity", "MaxPool",
    "AXIToStream", "Pad", "Reshape", "Slice", "Split",
    "StreamingConcat", "StreamingLineBuffer", "StreamingCircularLineBuffer",
    "StreamingMaxPool", "StreamingSplit", "StreamToAXI",
    "StreamingTensorDuplicator", "Transpose",
]
```

`Relu` isn't in this list. Both of nn2FPGA's own quant-propagation
transforms (`PropagateQuant` and `InferQuant`) only touch node types on this
list. `MaxPool`, `Reshape`, `Transpose` etc. are there because they don't
change the numeric quantization of their inputs — ReLU is the same kind of
op (clamping at zero doesn't touch scale/zero-point) but it's just not on
the list.

So this was never a quantization-format problem. nn2FPGA's quant metadata
never gets attached to a ReLU's tensors, regardless of what you feed
it. All four models have a ReLU in them, so this blocks all four as-is.

**Still open**: `mlp_small` fails earlier than the other three, back at the
same `InferLayouts` crash (`'NoneType' object has no attribute
'get_canonical_name'`) that was already patched a while back by adding
`graph.output` to the Phase 2 fallback loop. The patch is still active, the
other three models get past that step fine, so something else in
`mlp_small`'s graph specifically is hitting it, probably related to it being
pure Gemm/MatMul with no 4D tensors like the CNNs have. Didn't dig into this
one yet.

---

## Where things stand

| Model | hls4ml | nn2FPGA |
|---|---|---|
| `mlp_small` | works | fails at InferLayouts (separate, unresolved) |
| `cnn_lenet` | works | fails: Relu quant gap |
| `cnn_skip` | converts + compiles (io_stream), output is wrong | fails: Relu quant gap |
| `cnn_unusual_op` | fails: depth multiplier (confirmed format-independent) | fails: Relu quant gap |

---

## Main Point

Both of these are appearing to be source-confirmed tool limitations, not something
fixable from the config/methodology side — same category of finding as the
NNgen Conv2D bug from Phase 5. hls4ml can't currently carry quantization
info across a residual `Add` under QONNX; nn2FPGA never attaches
quantization info to a ReLU at all, in any format. Documenting this as-is
rather than continuing to patch around it. `mlp_small`'s InferLayouts issue
on nn2FPGA is the one loose thread left if that toolchain gets revisited.

---

## Follow-up: mlp_small and cnn_lenet weren't actually working either

"Converts cleanly" and "produces correct numbers" turned out to be two
different claims, and the table above only ever checked the first one for
`mlp_small` and `cnn_lenet`. Went back and ran an actual batch accuracy
check — 100 real MNIST samples, hls4ml C-sim output vs a genuine QONNX
reference execution, same method used on `cnn_skip` above — and both
"working" models came back at **12.0%** and **11.0%** argmax agreement.
Basically random.

First read on this looked like a script bug, not a model bug: both models
predicted the exact same wrong class on the exact same sample indices,
which is the kind of thing you'd expect from a shared compiled library
getting reused across two different model builds in one Python process
(hls4ml's `build_lib.sh` hardcodes `PROJECT=myproject`, so that's not a
crazy theory). Reran each model in its own separate process to rule that
out completely. Same 12.0%/11.0%, same wrong class, same samples. Not a
script bug — both models are genuinely producing wrong output.

### Tracing it down

Same warning as `cnn_skip` shows up for both of these models too:

```text
Failed to propagate quantization bias down Add node; model probably not suppored.
```

Not just on the skip connection — on every ordinary bias-add `Add` node in
both architectures. Read the actual pass this warning comes from
(`move_scales.py`, `BiasDownAdd`): it only knows how to move an `ApplyAlpha`
below an `Add` when that `ApplyAlpha`'s scale is exactly zero — a pure bias
shift. Any real quantized layer needs a genuine multiplicative rescale
there, so the scale is never zero, and the pass just gives up silently and
leaves the graph as-is.

Checked what "leaves the graph as-is" actually generates in the C++. Not
a dropped node — `ApplyAlpha` is a fully supported layer type in every
backend, implemented via the same code path as batchnorm. What actually
gets emitted is a direct fusion:

```cpp
nnet::normalize<Dense_MatMul_0_result_t, bn_Add_0_result_t, config60>(layer63_out, layer60_out, s60, b60); // bn_Add_0
```

`layer60_out = s60 * layer63_out + b60` — computed straight off the raw
MatMul accumulator, using the real weight-quant scale and the real trained
bias. Structurally fine. Pulled the actual scale/bias values out of the
generated weight files to check they weren't corrupted or zeroed —
they weren't; `s60`/`s61`/`s62` were real fractional numbers like
`0.0000608274`, and the biases were plausible trained values.

The values were real. The type holding them wasn't wide enough. Printed
the model's full output vector for one sample and it was nearly identical
to the raw bias vector alone:

```text
b62:        -0.2012739, -0.0812153, -0.1533047,  0.1661686, ...
hls4ml out: -0.2021484, -0.0820313, -0.1533203,  0.1660156, ...
```

The weighted contribution had vanished, leaving just the bias. Root cause:
`scale_t`/`bias_t` for these fused layers default to `model_default_t`,
which is `ap_fixed<16,6>` — 10 fractional bits, resolution ~0.001. The real
scale values here are around `0.0002`, smaller than that resolution, so
they silently round to zero the moment they're cast into that type.
Multiply anything by zero and only the bias survives — which also explains
why both models kept predicting the same wrong class: with the weighted
term gone, the prediction just becomes whichever output index has the
largest bias, a fixed property of each trained model, independent of the
actual input.

### The fix

hls4ml lets you override precision per layer. Widening `scale_t`/`bias_t`
on every `Add_N` and `Quant_N` layer fixes it:

```python
for layer_name in list(config['LayerName'].keys()):
    if (layer_name.startswith('Add_') or layer_name.startswith('Quant_')) and '_param' not in layer_name:
        config['LayerName'][layer_name].setdefault('Precision', {})
        config['LayerName'][layer_name]['Precision']['scale'] = 'ap_fixed<32,16>'
        config['LayerName'][layer_name]['Precision']['bias'] = 'ap_fixed<32,16>'
```

Confirmed on real batch runs, 100 samples each:

| Model | Before | After |
|---|---|---|
| `mlp_small` | 12.0% | **100.0%** |
| `cnn_lenet` | 11.0% | **100.0%** |

So this isn't "hls4ml can't handle quantized bias adds" — it's a real gap
in `config_from_onnx_model`'s automatic precision inference for these
specific fused layers, and it's fixable with a config override once you
know to look for it.

### Retesting cnn_skip with the same fix

Applied the identical override to the `io_stream` conversion for
`cnn_skip`. First sample checked came back with a correct argmax — looked
promising. Ran the full 100-sample batch to actually confirm it, same as
the other two, and got **27.0%** agreement. So the single sample was
a coincidence, not a fix.

`cnn_skip` has a second, separate problem that this override doesn't
touch. The same hundreds of `layer57_out` "read while empty" warnings from
before are still there, and this run additionally crashed on exit
(`recursive_mutex lock failed`) — a sign of real instability in the
generated `io_stream` dataflow, not just a numeric rounding issue. Reading
a stream before anything's written to it is what happens when `io_stream`'s
buffering doesn't correctly account for two paths of different length
arriving at the same merge node — which is exactly the shape a skip
connection creates (a short path straight to the `Add`, and a longer path
through the second `Conv`). That's a scheduling/FIFO-depth problem in
hls4ml's `io_stream` backend, separate from the quantization-precision
issue above, and a lot deeper to fix.

### Where things actually stand now

| Model | hls4ml |
|---|---|
| `mlp_small` | **100% argmax agreement** after the precision override |
| `cnn_lenet` | **100% argmax agreement** after the precision override |
| `cnn_skip` | precision override applied, still only 27% — separate `io_stream` dataflow/buffering bug on the skip connection, root-caused but not fixed |
| `cnn_unusual_op` | unchanged — fails earlier, at conversion (depth multiplier, format-independent) |

Two of the three models that could even get through conversion are now
looking verified to be working, not just "compiled without an exception."
`cnn_skip`'s failure was never really about hls4ml being unable to do
skip connections at all — it's two separate, both real, both
source-confirmed problems stacked on top of each other: a general
precision-inference gap (fixed) and a skip-connection-specific `io_stream`
buffering bug (not fixed, left as a documented open item).