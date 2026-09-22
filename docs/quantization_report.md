# Progress Update: Quantization & Toolchain Evaluation

## Summary

The same four models were ran through two quantization/deployment paths: QONNX
(arbitrary-scale INT8, feeding into hls4ml) and NNgen (power-of-two-scale
INT8, native ONNX import). QONNX handled all four models cleanly, 94-97%
accuracy across the board. NNgen only made it end-to-end on one model
(`mlp_small`), but where it worked, it matched QONNX almost exactly
(94.70% vs 94.8%). 


The project is moving forward with the QONNX path, feeding both hls4ml and
nn2FPGA, for the rest of the project. This not because NNgen's approach is
fundamentally worse, but rather it needed substantially more debugging
effort per model to get working on a benchmark, which doesn't make sense from an automation benchmark setup perspective

---

## Setup

- Host: macOS (Darwin arm64)
- Models: `mlp_small`, `cnn_lenet`, `cnn_skip`, `cnn_unusual_op`
- Eval set: 1,000-sample balanced slice
- Path A: QONNX arbitrary-scale INT8 PTQ
- Path B: NNgen power-of-two-scale INT8 PTQ

---

## Results

| Model | QONNX Acc | NNgen Acc | Notes |
|---|---|---|---|
| `mlp_small` | 94.8% | **94.70%** | Matches closely once the eval harness bug was fixed — real success, not a fluke |
| `cnn_lenet` | 96.5% | Failed (parse-time) | Reshape/Gemm shape mismatch, see below |
| `cnn_skip` | 97.5% | Failed (eval-time) | Confirmed independent of branching — see below |
| `cnn_unusual_op` | 96.8% | Failed (parse-time) | Grouped convolution not supported |

---

## Issues ran into for NNGen

**`cnn_lenet`** fails while parsing - inside NNgen's Gemm-import logicm 
specifically in the step where it tries to fuse a preceding Reshape node into
the following Gemm. It's expecting a 256-element flatten and getting 784.
The exact node responsible hasn't been traced yet, maybe worth a Netron pass if this
comes up again on a future model.

**`cnn_unusual_op`** fails on its grouped convolution - NNgen's ONNX Conv
importer doesn't handle the `group` attribute correctly, and throws a
channel-mismatch error. This looks like a confirmed operator gap.

**`cnn_skip`** was had some variations. First pass looked like a
branching/skip-connection problem — the error showed up nested inside
what looked like the residual merge. Built a minimal isolation test:
one plain Conv2d, no branching, vs. the same conv with a trivial skip
connection added. **Both failed identically, at the same line.** So it's
not branching rather it's that NNgen's software eval path (`verify/conv2d.py`)
breaks on Conv2d in general in this setup, regardless of architecture. The
shape mismatch (9 vs 252) seems like a patch-extraction bug, possibly
padding-related, though not confirmed against the actual source.

Another note: NNgen's release history is relatively old (no major activity in a
few years), and one export run hit an opset-downgrade failure
specific to a newer PyTorch ONNX exporter. Combined with the Conv2d bug,
the likely issue is a version-compatibility gap between this NNgen release
and current PyTorch/ONNX export tooling, not a fundamental flaw in NNgen's
design.

## Reason for moving forward with QONNX based toolchains

Not "QONNX is better." It's that getting NNgen working per-model required
real, tool-internal debugging (wrong API assumptions, digging into
tracebacks several layers deep, building isolation tests) that doesn't
scale well if the goal is a reasonably automated PyTorch-to-FPGA pipeline
across arbitrary architectures. QONNX's explicit `Quant` operator keeps
quantization info in the graph itself and plugged into the other toolchains with much
less friction.

If NNgen is revisited later — a newer release, or with more time to deal with
the Conv2d bug down to its actual source — the `mlp_small` result suggests
there's nothing fundamentally wrong with its approach once the pipeline
around it actually works.