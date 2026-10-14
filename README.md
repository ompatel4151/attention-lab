# Attention Lab

A small, reproducible project that checks scaled dot-product attention for correctness and
benchmarks inference latency. It compares a readable PyTorch reference implementation with
PyTorch's fused `torch.nn.functional.scaled_dot_product_attention` (SDPA).

**What this is:** a correctness harness and a careful micro-benchmark on synthetic tensors.
**What this is not:** a custom CUDA kernel, a transformer, or a claim about LLM performance.
Calling PyTorch's fused operator is *using* someone else's kernels, not writing one.

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
.venv/bin/python -m pytest -q                                   # correctness + schema tests
.venv/bin/python -m attention_lab.benchmark --device cpu --preset smoke --output results/smoke.json
.venv/bin/python -m attention_lab.benchmark --device cpu --preset cpu-sweep --output results/cpu-sweep.json
.venv/bin/python -m attention_lab.report                        # reports/report.md from results/*.json
```

Benchmark options: `--device {cpu,cuda,auto}`, `--preset {smoke,cpu-sweep,cuda-sweep}`, and overrides
`--batch --heads --head-dim --seq-lens --dtypes --causal {both,on,off} --warmup --min-run-time --threads`.
Run `--help` for the full list.

## Layout

```
src/attention_lab/attention.py   reference_attention, sdpa_attention, causal_mask, validate_qkv
src/attention_lab/benchmark.py   CLI: timing, correctness vs float64, environment record, JSON/CSV
src/attention_lab/report.py      Markdown table + matplotlib plots from saved JSON only
tests/                           test_attention.py, test_benchmark.py, test_report.py
results/                         raw runs (JSON with every timing sample, CSV summary)
reports/                         generated report.md + PNGs
docs/design.md                   shapes, masking, autograd vs inference, numerical error
docs/benchmark_method.md         timing boundary, warmup, sync, memory, pitfalls, limitations
docs/interview.md                questions, explain-without-reading checklist, resume templates
```

## Results (Apple M4 CPU, 4 threads, float32, B=1, H=2, D=32)

These are median latencies per call. Each range spans 3 independent runs of the `cpu-sweep` preset
(`results/cpu-sweep*.json`). The full table, IQRs and raw samples are in
[reports/report.md](reports/report.md).

| L | mask | reference (µs) | SDPA (µs) | reference / SDPA |
|---|---|---|---|---|
| 32 | noncausal | 24.5–24.6 | 17.2–17.5 | 1.41–1.43× |
| 32 | causal | 39.1–39.3 | 14.9–15.1 | 2.59–2.64× |
| 256 | noncausal | 90.4–91.0 | 45.9–47.0 | 1.94–1.97× |
| 256 | causal | 163.9–164.6 | 60.1–60.3 | 2.72–2.74× |

Both methods matched a float64 reference to within 6.7e-7 max absolute error in every configuration.

![latency vs sequence length](reports/cpu-sweep-latency.png)

How to read this:
- It is one isolated attention call at small sizes on one laptop CPU. It is not LLM performance.
- SDPA's backend was chosen automatically and not verified.
- Latency grew 3–4× while L grew 8×, so fixed per-call overhead still matters a lot at these sizes.
- The causal reference is slow partly because it rebuilds its [L, L] mask on every call.

## Status

- CPU: 35 tests pass. Benchmarks were run 3 times and the report was generated from those saved runs.
- CUDA: the code paths (CUDA events, fp16/bf16, peak memory) exist but **have not been run**. No NVIDIA
  GPU was available. fp16/bf16 tolerances are unvalidated guesses until someone runs them.

## AI assistance

The code, tests and docs were drafted with an AI pair programmer (Claude Code), one milestone at a
time. All numbers in `results/` come from real runs on the machine recorded in each JSON file.
Author's own contributions: [describe the change(s) you made yourself and the tests you added].

## Attribution

- Attention formula: Vaswani et al., "Attention Is All You Need", 2017 (arXiv:1706.03762).
- SDPA semantics: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Tiled attention (why fused kernels can avoid storing the L×L matrix): Dao et al., "FlashAttention", 2022 (arXiv:2205.14135).
- Timing utilities: https://pytorch.org/docs/stable/benchmark_utils.html
