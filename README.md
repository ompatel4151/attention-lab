# Attention Lab

By [ompatel4151](https://github.com/ompatel4151).

A small, reproducible project that checks scaled dot-product attention for correctness and
benchmarks it for inference on CPU and Apple GPU (MPS). It compares:

- **reference**: the textbook formula in PyTorch tensor ops, which materialises the [L, L] score matrix.
- **tiled**: the FlashAttention online-softmax recurrence in plain PyTorch. It loops over key blocks, never
  allocates an L×L matrix, and for causal attention skips the work above the diagonal.
  **tiled_compiled** is the same loop with each block step fused by `torch.compile`.
- **sdpa**: PyTorch's fused `scaled_dot_product_attention`, including variants forced to one backend
  with `sdpa_kernel`. A profiler records which kernel actually ran.

**What this is not:** a custom CUDA/Triton kernel, a transformer, or a claim about LLM performance.
`tiled` implements the *algorithm* behind FlashAttention, written in Python over PyTorch ops.
It is not a fused kernel. Calling SDPA uses PyTorch's kernels; it doesn't write one.

## Install (tested on macOS / Apple M4: CPU and MPS)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
```

`tiled_compiled` uses `torch.compile`, which needs a C++ compiler on CPU (on macOS: `xcode-select --install`).

Without `uv` (not tested): `python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
macOS: the default PyPI `torch` wheel (CPU + MPS) is what was tested. Linux CPU-only (not tested):
the default Linux wheel bundles CUDA libraries, so install from the CPU index instead with
`--index-url https://download.pytorch.org/whl/cpu`. Install a CUDA build only on a machine that has
an NVIDIA GPU (see https://pytorch.org/get-started/locally/).

Tested versions: Python 3.12.13, torch 2.14.1, pytest 9.1.1, matplotlib 3.11.2, on macOS 27.0.1 (arm64).

## Use

```bash
.venv/bin/python -m pytest -q                                   # 82 tests (MPS tests skip without Apple GPU)
.venv/bin/python -m attention_lab.benchmark --preset smoke --output results/smoke.json
.venv/bin/python -m attention_lab.benchmark --preset cpu-sweep --output results/cpu-sweep.json
.venv/bin/python -m attention_lab.benchmark --preset cpu-large --methods reference sdpa tiled \
    tiled_compiled sdpa_math sdpa_flash sdpa_efficient --output results/cpu-large.json
.venv/bin/python -m attention_lab.benchmark --preset mps-sweep --methods reference sdpa tiled \
    tiled_compiled sdpa_math sdpa_flash --output results/mps-sweep.json
.venv/bin/python -m attention_lab.report                        # reports/report.md from results/*.json
```

Options: `--device {cpu,cuda,mps,auto}`, `--preset {smoke,cpu-sweep,cpu-large,mps-sweep,cuda-sweep}`,
`--methods`, `--block-size` (tiled), `--rounds` (interleaved timing rounds), and the overrides
`--batch --heads --head-dim --seq-lens --dtypes --causal --warmup --min-run-time --event-reps --threads`.
Run `--help` for the full list. The profiler prints `USDT ... profiler_start/stop` lines to stderr;
they're harmless.

## Layout

```
src/attention_lab/attention.py   reference, tiled (+ compiled), sdpa wrappers; causal_mask, validate_qkv
src/attention_lab/benchmark.py   CLI: interleaved timing (CPU Timer / GPU events), bootstrap CIs, correctness vs
                                 CPU float64, profiler (kernel + CPU memory), JSON/CSV
src/attention_lab/report.py      Markdown tables + latency/memory plots + computed observations, from saved JSON only
tests/                           test_attention.py (69), test_benchmark.py (10), test_report.py (3)
results/                         raw runs: JSON with every timing sample, CSV summary
reports/                         generated report.md + PNGs
docs/                            writeup.md (start here), design.md, benchmark_method.md, interview.md
```

## Results

Apple M4, torch 2.14.1. CPU: 4 threads, float32, B = 1, H = 2, D = 32, three independent runs per preset
(ranges below span the runs). MPS: B = 1, H = 4, D = 64, one run. Full tables with 95% confidence intervals,
raw samples and plots are in [reports/report.md](reports/report.md). The two-page summary is
[docs/writeup.md](docs/writeup.md).

Every method matched a float64 reference in every configuration. Max absolute error: ≤ 8.9e-7 on CPU
float32, ≤ 2.2e-6 on MPS float32, ≤ 2.3e-3 for float16 (tolerance 1e-2) and ≤ 1.8e-2 for bfloat16
(tolerance 5e-2) on MPS.

**CPU memory at L = 4096** (peak above baseline during one call; identical in every run):

| method | noncausal | causal | growth L = 512 → 4096 (8×) |
|---|---|---|---|
| reference | 257 MiB | 272 MiB | ×62–64 (L² = ×64) |
| sdpa_math (forced) | 289 MiB | 353 MiB | ×62–63 |
| **tiled** | **8.2 MiB** | **8.2 MiB** | **×8 (linear)** |
| tiled_compiled | 4.2 MiB | 8.0 MiB | ×8 noncausal, ×15 causal |
| sdpa (auto) = sdpa_flash | 3.2 MiB | 3.2 MiB | ×4.8 |

![CPU memory vs L](reports/cpu-large-memory.png)

**CPU latency at L = 4096** (median per call; ref/x > 1 = faster than the reference):

| method | noncausal | ref/x | causal | ref/x |
|---|---|---|---|---|
| reference | 21.5–24.5 ms | 1 | 27.8–33.6 ms | 1 |
| **tiled** | 22.6–27.5 ms | 0.89–1.00× | **15.8–17.6 ms** | **1.74–1.90×** |
| tiled_compiled | 20.1–27.2 ms | 0.90–1.12× | 16.5–20.0 ms | 1.65–1.69× |
| sdpa_math (forced) | 24.4–27.1 ms | 0.87–0.93× | 30.0–38.7 ms | 0.87–0.93× |
| sdpa (auto, flash kernel) | 6.4–7.0 ms | 3.21–3.69× | 4.9–5.3 ms | 5.36–6.39× |

At small L the picture flips: at L = 256 tiled is 0.29× (noncausal) and 0.50–0.53× (causal) the reference's
speed, while SDPA is 1.97–1.98× and 2.71–2.76×.

![CPU latency vs L](reports/cpu-large-latency.png)

**MPS (Apple GPU) at L = 4096, one run** (ref/x with its 95% CI):

| dtype | mask | reference | sdpa | tiled |
|---|---|---|---|---|
| float32 | noncausal | 22.3 ms | 7.5 ms (2.98× [2.95, 3.02]) | 65.5 ms (0.34× [0.32, 0.34]) |
| float32 | causal | 32.1 ms | 4.0 ms (8.11× [7.73, 8.33]) | 82.3 ms (0.39× [0.37, 0.40]) |
| float16 | causal | 17.8 ms | 3.1 ms (5.79× [5.59, 5.88]) | 74.3 ms (0.24× [0.23, 0.24]) |
| bfloat16 | causal | 18.0 ms | 3.1 ms (5.80× [5.59, 5.89]) | 71.4 ms (0.25× [0.25, 0.26]) |

**What this shows:**
- **Tiling trades a little speed for a lot of memory, and wins on causal at large L.** At L = 4096 tiled
  uses 31× less memory than the reference. Causal tiled is 1.7–1.9× faster because it skips the triangle
  above the diagonal, while the reference computes the full matrix and then masks it.
- **Finding what to fix took a profile.** The first causal version did no better than noncausal: building
  masks and concatenating state every block cancelled the halved `exp`/matmul work. Masking only the
  diagonal tile and emitting finished rows early fixed it.
- **`torch.compile` helps memory more than speed.** Compiling the per-block step fuses ops and halves tiled's
  noncausal peak memory (4.2 vs 8.2 MiB). Its speedup isn't distinguishable from eager across runs.
- **SDPA's CPU speed comes from its flash kernel.** The profiler saw `_scaled_dot_product_flash_attention_for_cpu`.
  Forcing MATH (`_scaled_dot_product_attention_math`) is slower than our reference and uses more memory.
  `EFFICIENT_ATTENTION` has no CPU kernel and is recorded as skipped.
- **On MPS the Python loop loses.** Each of tiled's ~20 ops per block is a separate GPU kernel launch, so at
  L = 4096 it runs at 0.24–0.39× the reference's speed. Automatic SDPA, forced MATH and forced FLASH all ran the same
  `_scaled_dot_product_attention_math_for_mps` kernel there.

**Limits:** one laptop (Apple M4), batch 1, inference only. CPU medians varied across runs by a median of
11% (up to 35%), which is wider than the within-run CIs, so treat single-run CIs as optimistic. MPS has no
peak-memory counter, so MPS memory is not reported. CUDA paths exist but have never run. Nothing here is
LLM performance.

## Status

- 82 tests pass on CPU and MPS. CPU presets were run 3×, MPS once; the report is generated from those saved runs.
- CUDA: the code paths (CUDA events, peak memory) exist but **have not been run**. No NVIDIA GPU was
  available, so fp16/bf16 tolerances are validated on MPS only.

## Attribution

- Attention formula: Vaswani et al., "Attention Is All You Need", 2017 (arXiv:1706.03762).
- Online softmax and tiling: Dao et al., "FlashAttention", 2022 (arXiv:2205.14135); Milakov & Gimelshein,
  "Online normalizer calculation for softmax", 2018 (arXiv:1805.02867).
- SDPA semantics: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Backend selection: https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
- Timing and profiling: https://pytorch.org/docs/stable/benchmark_utils.html, https://pytorch.org/docs/stable/profiler.html
