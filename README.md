# Bit-exact MTP self-speculative decoding for Qwen3.5 on Apple Silicon (MLX)

Greedy decoding of Qwen3.5-9B (4-bit) on a 16 GB M1 Pro: 48.3 tok/s on a 20-prompt suite (1.38x stock mlx-lm, 1.58x on math; 2.1x llama.cpp b10969; 4B 1.32x, 2B 1.24x), with
output **byte-identical** to plain greedy MLX decoding. The paper draft is in `paper/`.

## Use it

```sh
pip install -r requirements.txt
python fetch_mtp.py                       # 15 MTP tensors of Qwen/Qwen3.5-9B by HTTP range request (~0.5 GB)
python fastqwen.py "your prompt" --think  # streams the answer and prints tok/s
python fastqwen.py "..." --fast           # allow 3-token drafts: faster on easy text, may flip exact logit ties
```

The MLX model is `mlx-community/Qwen3.5-9B-4bit`; set the path in `fastqwen.py` / `--model`.
Other sizes: `python fetch_mtp.py Qwen/Qwen3.5-4B models/mtp4b/mtp.safetensors` and `--model/--mtp` on the benchmarks.

## How it works (file map)

| file | role |
|---|---|
| `specmtp.py` | MTP head (`load_mtp`), trimmed-vocabulary draft head (`Head`, `frequent_vocab`), reference speculative loop |
| `specpipe.py` | production loop: optimistic prebuild of the next cycle; restores KV *views* on a miss (MLX in-place-write hazard) |
| `specmlx.py` | capture-mode Gated DeltaNet (per-position state snapshots) + `rewind`; plain greedy reference |
| `qmvt.py` | multi-token 4-bit mat-vec Metal kernel (T=2..4); T=2/3 bit-exact vs MLX's single-token kernel |
| `specfast.py` | `mx.compile` of capture-mode DeltaNet layers (cuts per-cycle CPU graph build) |
| `fusedgdn.py` | one Metal kernel for each DeltaNet layer's middle (conv, norms, gates, recurrence, gated norm), bit-exact vs MLX; `python fusedgdn.py` checks it |
| `fetch_mtp.py` | range-request download of MTP tensors |
| `fastqwen.py` | command-line generator |

## Reproduce the paper's numbers

| result | command |
|---|---|
| baselines (llama.cpp / MLX) | `bench.py NAME [--engine mlx] [-- flags]` |
| T-scaling, kernel comparison | `kernel_compare.py` (includes a port of MLX's `qmv_wide`, `qmvwide.py`) |
| main table (20-prompt suite, CIs) | `final_bench.py --tag 9b_fused --fused-gdn` (+ `--model/--mtp ... --llamacpp` for 4B/2B), `llama_suite.py 9b` |
| long reasoning traces | `longrun.py` |
| quality vs llama.cpp Q4_K_M (paired McNemar) | `cd quality && python compare.py 100 math_dev,gsm_dev mlx_spec` (`KEY=ok2048` for the 2048 cap); raw outputs in `quality/results/` |
| quality (GSM8K/MATH, exact mode) | `quality_spec.py math_dev 100`, `quality_spec.py gsm_dev 100` |
| ablations | `specmtp.py -k 1 0 [--full-head] [--kernel mlx] [--compile] [--pipe] [--vocab N]` |
| fused DeltaNet kernel (paired A/B, micro-benchmark) | `fused_bench.py`, `gdn_micro.py`, `gdn_ablate.py` |
| where the non-matmul time goes | `firstprinciples.py`, `where.py` |
| profiling | `profile_spec.py`, `profile_forward.py`, `profile_layers.py`, `profile_cycle.py` |
| figures | `paper/make_figures.py` |

Negative-result code is kept for reference: `fuse.py` (projection fusion), `fastdecode.py` (compiled T=1 step),
`mmaqmv*.py` (simdgroup-matrix kernels), `fastqmv.py` (first scalar kernel), `specmlx.py` (0.8B draft model).

All measurements: macOS 14.7, MLX 0.32.4, M1 Pro 16 GB, Low Power Mode off. Each benchmark waits for a
nominal thermal state; compare configurations within one session (absolute speed drifts by up to ~8%).
