# Attention Lab

By [ompatel4151](https://github.com/ompatel4151).

A small, reproducible project that checks scaled dot-product attention for correctness and
benchmarks it for inference. It compares three implementations:

- **reference**: the textbook formula in PyTorch tensor ops, which materialises the [L, L] score matrix.
- **tiled**: the FlashAttention online-softmax recurrence in plain PyTorch. It loops over key blocks
  and never allocates an L×L matrix.
- **sdpa**: PyTorch's fused `scaled_dot_product_attention`, including variants forced to one backend
  with `sdpa_kernel`. A profiler records which kernel actually ran.

**What this is not:** a custom CUDA/Triton kernel, a transformer, or a claim about LLM performance.
`tiled` implements the *algorithm* behind FlashAttention, written in Python over PyTorch ops.
It is not a fused kernel. Calling SDPA uses PyTorch's kernels; it doesn't write one.

## Install (tested on macOS / Apple M4, CPU only)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
```

Without `uv` (not tested): `python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
macOS: the default PyPI `torch` wheel (CPU + MPS) is what was tested. Linux CPU-only (not tested):
the default Linux wheel bundles CUDA libraries, so install from the CPU index instead with
`--index-url https://download.pytorch.org/whl/cpu`. Install a CUDA build only on a machine that has
an NVIDIA GPU (see https://pytorch.org/get-started/locally/).

Tested versions: Python 3.12.13, torch 2.14.1, pytest 9.1.1, matplotlib 3.11.2, on macOS 27.0.1 (arm64).

## Use

```bash
.venv/bin/python -m pytest -q                                   # 77 tests
.venv/bin/python -m attention_lab.benchmark --preset smoke --output results/smoke.json
.venv/bin/python -m attention_lab.benchmark --preset cpu-sweep --output results/cpu-sweep.json
.venv/bin/python -m attention_lab.benchmark --preset cpu-large \
    --methods reference sdpa tiled sdpa_math sdpa_flash sdpa_efficient --output results/cpu-large.json
.venv/bin/python -m attention_lab.report                        # reports/report.md from results/*.json
```

Options: `--device {cpu,cuda,auto}`, `--preset {smoke,cpu-sweep,cpu-large,cuda-sweep}`, `--methods`,
`--block-size` (tiled), and the overrides `--batch --heads --head-dim --seq-lens --dtypes --causal --warmup
--min-run-time --threads`. Run `--help` for the full list. The profiler prints `USDT ... profiler_start/stop`
lines to stderr; they're harmless.

## Layout

```
src/attention_lab/attention.py   reference_attention, tiled_attention, sdpa_attention, causal_mask, validate_qkv
src/attention_lab/benchmark.py   CLI: timing, correctness vs float64, profiler (kernel + CPU memory), JSON/CSV
src/attention_lab/report.py      Markdown tables + latency/memory plots + computed observations, from saved JSON only
tests/                           test_attention.py (67), test_benchmark.py (8), test_report.py (2)
results/                         raw runs: JSON with every timing sample, CSV summary
reports/                         generated report.md + PNGs
docs/                            design.md, benchmark_method.md, interview.md
```

## Results (Apple M4 CPU, 4 threads, float32, B=1, H=2, D=32)

Each range covers 3 independent runs. Full tables, IQRs, raw samples and plots are in
[reports/report.md](reports/report.md). Every method matched a float64 reference within 8.9e-7
max absolute error in every configuration.

**Memory** (`cpu-large`, L = 4096, peak bytes above baseline during one call; identical in all runs):

| method | noncausal | causal | growth from L = 512 to 4096 (8×) |
|---|---|---|---|
| reference | 257 MiB | 272 MiB | ×62 (L² = ×64) |
| sdpa_math (forced) | 289 MiB | 353 MiB | ×62 |
| **tiled (ours)** | **9.2 MiB** | **9.2 MiB** | **×8 (linear)** |
| sdpa (auto) = sdpa_flash | 3.2 MiB | 3.2 MiB | ×4.8 |

![memory vs L](reports/cpu-large-memory.png)

**Latency** (median per call; ref/x > 1 means faster than the reference):

| L | mask | reference | tiled (ref/tiled) | sdpa auto (ref/sdpa) | sdpa_math (ref/math) |
|---|---|---|---|---|---|
| 32 | noncausal | 24.0–26.4 µs | 28.1–28.5 µs (0.85–0.92×) | 16.8–17.7 µs (1.42–1.51×) | not run |
| 256 | noncausal | 89.2–90.2 µs | 309–319 µs (0.28–0.29×) | 45.7–51.2 µs (1.76–1.96×) | not run |
| 4096 | noncausal | 24.2–27.6 ms | 27.0–31.0 ms (0.85–1.01×) | 7.1–8.2 ms (3.37–3.61×) | 27.0–31.2 ms (0.88–0.94×) |
| 4096 | causal | 33.4–34.9 ms | 32.0–33.6 ms (1.00–1.04×) | 5.4–5.9 ms (5.91–6.28×) | 34.5–37.8 ms (0.92–0.98×) |

![latency vs L](reports/cpu-large-latency.png)

**What this shows:**
- **Tiled trades speed for memory, then reaches parity.** Its Python loop launches about 18–21 small aten ops
  per key block (counted with the profiler), so it is about 3.5× slower than the reference at L = 256. By L = 4096 it roughly matches the
  reference's latency while using 28× less memory.
- **SDPA's speedup is the flash kernel, not "SDPA".** The profiler saw
  `aten::_scaled_dot_product_flash_attention_for_cpu` for automatic SDPA. Forcing the MATH backend
  (`aten::_scaled_dot_product_attention_math`) is no faster than our reference and uses more memory.
  `EFFICIENT_ATTENTION` has no CPU kernel, so it is recorded as skipped with PyTorch's error.
- **Causal gets cheaper only for the flash kernel** (5.4–5.9 ms causal vs 7.1–8.2 ms noncausal at L = 4096).
  That's consistent with it skipping fully masked blocks. The reference, MATH and tiled all compute the
  full matrix and then mask it.
- **At small L, overhead dominates.** From L = 32 to 256 the reference's latency grew 3–4× while L² grew 64×.

**Limits:** one laptop CPU, batch 1, two heads, float32, inference only. In `cpu-large`, the same
configuration's median varied by a median of 14% across runs (5–16% at L = 4096, up to 73% at L = 1024),
so treat ratios as ranges, not point values. Nothing here is
GPU or LLM performance.

## Status

- CPU: 77 tests pass. Benchmarks were run 3× per preset, and the report was generated from those saved runs.
- CUDA: the code paths (CUDA events, fp16/bf16, peak memory) exist but **have not been run**. No NVIDIA
  GPU was available. fp16/bf16 tolerances are unvalidated guesses.

## Attribution

- Attention formula: Vaswani et al., "Attention Is All You Need", 2017 (arXiv:1706.03762).
- Online softmax and tiling: Dao et al., "FlashAttention", 2022 (arXiv:2205.14135); Milakov & Gimelshein,
  "Online normalizer calculation for softmax", 2018 (arXiv:1805.02867).
- SDPA semantics: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Backend selection: https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
- Timing and profiling: https://pytorch.org/docs/stable/benchmark_utils.html, https://pytorch.org/docs/stable/profiler.html
