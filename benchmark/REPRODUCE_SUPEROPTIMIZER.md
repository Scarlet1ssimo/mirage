# Reproduce the academic Mirage superoptimizer

This runner reproduces the A100 experiments on the `evaluation` compiler,
including timing candidates whose outputs contain NaN/Inf or fail the numerical
thresholds. It uses the **existing Forest timer/numerical metrics and XLA
fixtures**, rather than introducing another reference implementation.
It is based on upstream `e8980e19` plus this fork's `2c0ae68d` installation fixes.
MPK is not used.

## Environment

Use the existing CUDA/PyTorch environment and Forest/XLA checkouts. Run from the
Mirage repository. The tested A100 environment was Python 3.10.21,
Torch 2.7.0+cu126, upstream Triton 3.3.0, nvcc 12.8.61, z3-solver 4.16.0.0.
The worker detects SM80/SM90 from the selected GPU. H100 has not been tested here.

```bash
# Load the CUDA module available on your machine; module name is site-specific.
module avail cuda
module load <cuda-module>
export MIRAGE_ROOT="$PWD"
export PYTHONPATH="$MIRAGE_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
git submodule update --init deps/cutlass deps/json

# Build for the actual GPU (80 for A100, 90 for H100).
export MIRAGE_CUDA_ARCHITECTURES=90
python setup.py build_ext --inplace --force
```

The fork's setup.py retains its CUDA 12.8 include/library paths. If your site
uses another installation location, adjust those **build paths only**, or use
its existing Mirage installation environment. No optimizer changes are needed.
The search checkpoint fix saves each verified candidate atomically so that a
bounded search can be stopped without losing completed candidates.

## Original Forest shapes: ten cases

`--forest-root` must contain `benchmarks/src/forest_benchmarks/timing.py` and
`codegen/src/forest/codegen/numerics.py`. `--fixture-dir` is an existing XLA output
root with `<case>/B{1,16}/inputs.npz`, `reference.npy`, and `settings.json`.
The case folders are `attention`, `gqa-attention`, `layernorm-matmul`,
`rms-ffn-swiglu`, and `efficient-vit`. Use fixtures with the same seed and shapes
as the Forest comparison; this runner does not regenerate them.

```bash
python benchmark/reproduce_superoptimizer.py \
  --suite forest --source fresh --gpus 2 3 \
  --forest-root /path/to/forest --fixture-dir /path/to/xla-eval/outputs/run \
  --fold-zero-epsilon --allow-invalid-output \
  --output outputs/h100-forest-fresh
```

The GPU IDs above are examples: use only GPUs allocated to your experiment.
Attention is native MHA `[B,32,4096,128]`, not the historical GQA cosplay.
GQA has 32 query / 8 KV heads; LN has X `[B,8192]`, W `[8192,28672]`;
the full FFN has X `[B*4096,96]`, W/V `[96,192]`, U `[192,96]`;
EfficientViT has Q/K/V `[B,128,128]`. B defaults to 1 and 16.
`--fold-zero-epsilon` removes EfficientViT's `1e-15` scalar after FP16
underflow; it is labeled in the output, and the fixture reference is unchanged.

## AE shapes and caches

```bash
# Reproduce our ordinary-cache measurements, including numerical failures.
python benchmark/reproduce_superoptimizer.py \
  --suite ae --source cache --cache-kind ordinary --gpus 2 3 \
  --forest-root /path/to/forest --allow-invalid-output \
  --output outputs/h100-ae-ordinary

# H100-specific caches are a separate experiment.
python benchmark/reproduce_superoptimizer.py \
  --suite ae --source cache --cache-kind hopper --gpus 2 3 \
  --forest-root /path/to/forest --allow-invalid-output \
  --output outputs/h100-ae-hopper

# Search without consuming an author-provided checkpoint.
python benchmark/reproduce_superoptimizer.py \
  --suite ae --source fresh --gpus 2 3 \
  --forest-root /path/to/forest --allow-invalid-output \
  --output outputs/h100-ae-fresh
```

AE defaults reproduce MHA B=1/8, GQA B=1/16, RMSNorm+Matmul B=1/16,
and GatedMLP with a fixed eight input rows. These are different formulas/shapes
from the Forest suite: AE attention is unscaled, RMSNorm has a single matmul,
and GatedMLP has neither RMSNorm nor an output projection. The RMSNorm runtime
weight uses the graph's correct `[4096,6144]` shape. Missing cache files are
reported as `cache_missing`; there is no implicit B=16 to B=8 or Hopper to
ordinary fallback. In particular, the supplied Hopper cache directory has no
MHA cache, and the ordinary cache has no MHA B=16.

To repeat a specific configuration, use `--cases`, `--batch-sizes`, and
`--candidate-indices` (zero-based). Our A100 representative indices were:

| AE case | B | CUDA candidate indices |
|---|---:|---|
| attention | 1 | 0 |
| attention | 8 | 18, 23, 28 |
| gqa | 1 | 0, 7, 14 |
| gqa | 16 | 0, 8, 14 |
| rmsnorm | 1 | 0, 3, 8, 17 |
| rmsnorm | 16 | 0, 1, 10, 21 |
| gated_mlp | 1 | 0, 9, 17 |

For example, append `--cases gqa --batch-sizes 16 --candidate-indices 0 8 14`.
Indices not present in the selected checkpoint are omitted; the chosen subset
and any truncation are recorded. `--max-candidates N` selects a prefix instead.

## Optional precision adapter and Triton

`--fp32-accum` adapts **generated CUDA only**: SM80 MMA accumulation and
`CUBLAS_COMPUTE_16F` become FP32 accumulation while input/output remain FP16.
Run it as a separately labeled experiment. It rejects Hopper WGMMA kernels and
kernels without supported replacement sites, rather than claiming an adaptation
that did not occur. Both original and adapted CUDA are exported. A100 validated
RMSNorm candidates were B=1 index 3 and B=16 index 1; GatedMLP used index 0.

Use `--backends cuda triton` to attempt both backends. Evaluation-era generated
Triton uses `triton.ops`, absent in modern Triton. If needed, install upstream
Triton 2.3.1 into an isolated directory and pass `--triton-path /path/to/legacy`;
this does not replace the existing Forest environment. Compilation/runtime
failures remain in the report. CUDA is sufficient for the primary reproduction.

## Protocol and artifacts

Defaults: seed 1729, explicit warmup 10, `do_bench` with L2 flushing and 200ms
measurement, thermal cooldown, search threads 8, search 180s, each candidate
180s, and each case 600s. Change budgets with `--search-budget-s`,
`--candidate-budget-s`, and `--case-budget-s`; the case budget includes search.
CUDA uses preallocated scratch/output and default pipeline stages=2,
warp groups=2. Alternative settings are explicit CLI options, not an additional
autotuning sweep. The fastest result is only the fastest among timed candidates.

Only explicitly selected physical GPU IDs/UUIDs are used, one sequential worker
per GPU, with a per-device lock. Output directories must be empty. Ctrl-C stops
owned process groups and exports the completed diagnostics.

- `performance.csv`: one row per case, with **validated** performance and
  **fastest timed** performance in separate columns; NaN/Inf counts and errors
  are included. Invalid output never becomes a validated winner.
- `environment.json`: GPU model, UUID, driver, power cap, compiler, package
  versions, commits, command arguments, and protocol. Per-attempt `result.json`
  adds the actual device architecture, imported software, and timing power samples.
- `attempts.json`, `failure-distribution.json`, and each `run.log`: failed,
  crashed, missing-cache, and timeout diagnostics.
- `cases/<case>/B*/winner/` and `fastest_timed/`: candidate JSON μGraph, generated
  CUDA/Python, actual tuning, provenance through command/result files, hashes,
  and Input IR when fresh search/direct export supplied it. These are Mirage
  JSON graphs, not Forest protobufs.

The worker records numerical metrics before and after timing. Without
`--allow-invalid-output`, failed validation stops before measurement. With it,
failed output is labeled `measured_invalid`; error entries are JSON null for
NaN/Inf, never zero. Numerical thresholds remain relative L2 <0.005 and
normalized maximum error <0.02.
