# Attention Lab benchmark report

## `cpu-sweep-run2.json`

- Environment: cpu (arm64), torch 2.14.1, 4 CPU threads; Python 3.12.13; macOS-27.0.1-arm64-arm-64bit; CUDA runtime None; recorded 2026-10-14T05:54:17+00:00
- Shapes: B=1, H=2, D=32, L in [32, 64, 128, 256]; dtypes ['float32']; seed 0
- Timing: torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.2), timer=timeit.default_timer, num_threads=4; sample = mean per-call time of one block; 10 warmup calls per method; unit seconds
- Memory: unavailable on CPU
- SDPA backend: automatic selection; not verified
- Correctness: each method vs float64 reference; tolerances {'float32': {'atol': 1e-05, 'rtol': 1e-05}}

Reproduce (from the project root, inside the venv):

```bash
python -m attention_lab.benchmark --device cpu --preset cpu-sweep --output results/cpu-sweep-run2.json
```

| L | dtype | causal | reference µs (IQR) | SDPA µs (IQR) | ref/SDPA | max abs err ref / SDPA | correct |
|---|---|---|---|---|---|---|---|
| 32 | float32 | noncausal | 24.5 (0.6) | 17.4 (0.4) | 1.41x | 2.6e-07 / 2.9e-07 | yes |
| 32 | float32 | causal | 39.1 (1.3) | 14.9 (0.7) | 2.63x | 3.6e-07 / 3.1e-07 | yes |
| 64 | float32 | noncausal | 28.3 (0.3) | 18.6 (0.4) | 1.52x | 5.4e-07 / 3.4e-07 | yes |
| 64 | float32 | causal | 45.6 (1.4) | 19.3 (1.1) | 2.37x | 5.0e-07 / 2.5e-07 | yes |
| 128 | float32 | noncausal | 43.3 (0.8) | 25.2 (1.4) | 1.72x | 3.8e-07 / 3.6e-07 | yes |
| 128 | float32 | causal | 71.4 (1.3) | 27.2 (1.4) | 2.62x | 6.7e-07 / 4.9e-07 | yes |
| 256 | float32 | noncausal | 91.0 (2.0) | 47.0 (2.2) | 1.94x | 3.7e-07 / 3.5e-07 | yes |
| 256 | float32 | causal | 164.6 (4.3) | 60.1 (1.6) | 2.74x | 5.3e-07 / 4.7e-07 | yes |

![latency vs L](cpu-sweep-run2-latency.png)

### Observations (computed from this run)

- SDPA had the lower median latency in 8 of 8 configurations.
- float32, noncausal: reference/SDPA ratio 1.41x to 1.94x over L = 32, 64, 128, 256.
- float32, causal: reference/SDPA ratio 2.37x to 2.74x over L = 32, 64, 128, 256.

Latency growth across the measured range, compared with linear and quadratic growth in L:

| method | dtype | mask | L range | latency growth | L growth | L² growth |
|---|---|---|---|---|---|---|
| reference | float32 | noncausal | 32 → 256 | x3.7 | x8 | x64 |
| sdpa | float32 | noncausal | 32 → 256 | x2.7 | x8 | x64 |
| reference | float32 | causal | 32 → 256 | x4.2 | x8 | x64 |
| sdpa | float32 | causal | 32 → 256 | x4.0 | x8 | x64 |

How to read this:
- One run on one machine. Re-run before quoting numbers; compare only rows from the same environment.
- Timings cover one attention call including the identical Python validation in both wrappers;
  input creation, correctness checks and I/O are excluded. This is not whole-model performance.
- SDPA backend: automatic selection; not verified. Nothing here shows which kernel ran.
- The score matrix is O(L²), but at small L fixed per-call costs (Python, op dispatch, allocation)
  are a large share, so latency can grow much more slowly than L². Growth near L² is only expected
  once the O(L²·D) arithmetic dominates.
- The explicit causal reference builds an [L, L] mask and runs masked_fill on every call, which adds
  work the noncausal path does not do.

## `cpu-sweep-run3.json`

- Environment: cpu (arm64), torch 2.14.1, 4 CPU threads; Python 3.12.13; macOS-27.0.1-arm64-arm-64bit; CUDA runtime None; recorded 2026-10-14T05:54:22+00:00
- Shapes: B=1, H=2, D=32, L in [32, 64, 128, 256]; dtypes ['float32']; seed 0
- Timing: torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.2), timer=timeit.default_timer, num_threads=4; sample = mean per-call time of one block; 10 warmup calls per method; unit seconds
- Memory: unavailable on CPU
- SDPA backend: automatic selection; not verified
- Correctness: each method vs float64 reference; tolerances {'float32': {'atol': 1e-05, 'rtol': 1e-05}}

Reproduce (from the project root, inside the venv):

```bash
python -m attention_lab.benchmark --device cpu --preset cpu-sweep --output results/cpu-sweep-run3.json
```

| L | dtype | causal | reference µs (IQR) | SDPA µs (IQR) | ref/SDPA | max abs err ref / SDPA | correct |
|---|---|---|---|---|---|---|---|
| 32 | float32 | noncausal | 24.6 (0.5) | 17.5 (0.3) | 1.41x | 2.6e-07 / 2.9e-07 | yes |
| 32 | float32 | causal | 39.3 (1.4) | 14.9 (0.7) | 2.64x | 3.6e-07 / 3.1e-07 | yes |
| 64 | float32 | noncausal | 28.5 (0.6) | 18.7 (0.9) | 1.52x | 5.4e-07 / 3.4e-07 | yes |
| 64 | float32 | causal | 46.1 (1.6) | 19.3 (0.7) | 2.39x | 5.0e-07 / 2.5e-07 | yes |
| 128 | float32 | noncausal | 43.3 (0.9) | 25.2 (0.8) | 1.72x | 3.8e-07 / 3.6e-07 | yes |
| 128 | float32 | causal | 71.2 (1.3) | 27.2 (0.8) | 2.61x | 6.7e-07 / 4.9e-07 | yes |
| 256 | float32 | noncausal | 90.8 (2.1) | 46.1 (2.2) | 1.97x | 3.7e-07 / 3.5e-07 | yes |
| 256 | float32 | causal | 164.3 (5.9) | 60.3 (1.7) | 2.72x | 5.3e-07 / 4.7e-07 | yes |

![latency vs L](cpu-sweep-run3-latency.png)

### Observations (computed from this run)

- SDPA had the lower median latency in 8 of 8 configurations.
- float32, noncausal: reference/SDPA ratio 1.41x to 1.97x over L = 32, 64, 128, 256.
- float32, causal: reference/SDPA ratio 2.39x to 2.72x over L = 32, 64, 128, 256.

Latency growth across the measured range, compared with linear and quadratic growth in L:

| method | dtype | mask | L range | latency growth | L growth | L² growth |
|---|---|---|---|---|---|---|
| reference | float32 | noncausal | 32 → 256 | x3.7 | x8 | x64 |
| sdpa | float32 | noncausal | 32 → 256 | x2.6 | x8 | x64 |
| reference | float32 | causal | 32 → 256 | x4.2 | x8 | x64 |
| sdpa | float32 | causal | 32 → 256 | x4.1 | x8 | x64 |

How to read this:
- One run on one machine. Re-run before quoting numbers; compare only rows from the same environment.
- Timings cover one attention call including the identical Python validation in both wrappers;
  input creation, correctness checks and I/O are excluded. This is not whole-model performance.
- SDPA backend: automatic selection; not verified. Nothing here shows which kernel ran.
- The score matrix is O(L²), but at small L fixed per-call costs (Python, op dispatch, allocation)
  are a large share, so latency can grow much more slowly than L². Growth near L² is only expected
  once the O(L²·D) arithmetic dominates.
- The explicit causal reference builds an [L, L] mask and runs masked_fill on every call, which adds
  work the noncausal path does not do.

## `cpu-sweep.json`

- Environment: cpu (arm64), torch 2.14.1, 4 CPU threads; Python 3.12.13; macOS-27.0.1-arm64-arm-64bit; CUDA runtime None; recorded 2026-10-14T05:43:47+00:00
- Shapes: B=1, H=2, D=32, L in [32, 64, 128, 256]; dtypes ['float32']; seed 0
- Timing: torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.2), timer=timeit.default_timer, num_threads=4; sample = mean per-call time of one block; 10 warmup calls per method; unit seconds
- Memory: unavailable on CPU
- SDPA backend: automatic selection; not verified
- Correctness: each method vs float64 reference; tolerances {'float32': {'atol': 1e-05, 'rtol': 1e-05}}

Reproduce (from the project root, inside the venv):

```bash
python -m attention_lab.benchmark --device cpu --preset cpu-sweep --output results/cpu-sweep.json
```

| L | dtype | causal | reference µs (IQR) | SDPA µs (IQR) | ref/SDPA | max abs err ref / SDPA | correct |
|---|---|---|---|---|---|---|---|
| 32 | float32 | noncausal | 24.6 (0.4) | 17.2 (0.3) | 1.43x | 2.6e-07 / 2.9e-07 | yes |
| 32 | float32 | causal | 39.1 (0.6) | 15.1 (0.5) | 2.59x | 3.6e-07 / 3.1e-07 | yes |
| 64 | float32 | noncausal | 28.5 (0.4) | 18.7 (0.4) | 1.53x | 5.4e-07 / 3.4e-07 | yes |
| 64 | float32 | causal | 45.2 (3.9) | 19.6 (1.9) | 2.30x | 5.0e-07 / 2.5e-07 | yes |
| 128 | float32 | noncausal | 43.2 (1.9) | 25.9 (4.4) | 1.67x | 3.8e-07 / 3.6e-07 | yes |
| 128 | float32 | causal | 72.1 (1.2) | 27.1 (0.7) | 2.66x | 6.7e-07 / 4.9e-07 | yes |
| 256 | float32 | noncausal | 90.4 (1.5) | 45.9 (1.3) | 1.97x | 3.7e-07 / 3.5e-07 | yes |
| 256 | float32 | causal | 163.9 (4.0) | 60.1 (1.4) | 2.73x | 5.3e-07 / 4.7e-07 | yes |

![latency vs L](cpu-sweep-latency.png)

### Observations (computed from this run)

- SDPA had the lower median latency in 8 of 8 configurations.
- float32, noncausal: reference/SDPA ratio 1.43x to 1.97x over L = 32, 64, 128, 256.
- float32, causal: reference/SDPA ratio 2.30x to 2.73x over L = 32, 64, 128, 256.

Latency growth across the measured range, compared with linear and quadratic growth in L:

| method | dtype | mask | L range | latency growth | L growth | L² growth |
|---|---|---|---|---|---|---|
| reference | float32 | noncausal | 32 → 256 | x3.7 | x8 | x64 |
| sdpa | float32 | noncausal | 32 → 256 | x2.7 | x8 | x64 |
| reference | float32 | causal | 32 → 256 | x4.2 | x8 | x64 |
| sdpa | float32 | causal | 32 → 256 | x4.0 | x8 | x64 |

How to read this:
- One run on one machine. Re-run before quoting numbers; compare only rows from the same environment.
- Timings cover one attention call including the identical Python validation in both wrappers;
  input creation, correctness checks and I/O are excluded. This is not whole-model performance.
- SDPA backend: automatic selection; not verified. Nothing here shows which kernel ran.
- The score matrix is O(L²), but at small L fixed per-call costs (Python, op dispatch, allocation)
  are a large share, so latency can grow much more slowly than L². Growth near L² is only expected
  once the O(L²·D) arithmetic dominates.
- The explicit causal reference builds an [L, L] mask and runs masked_fill on every call, which adds
  work the noncausal path does not do.

## `smoke.json`

- Environment: cpu (arm64), torch 2.14.1, 4 CPU threads; Python 3.12.13; macOS-27.0.1-arm64-arm-64bit; CUDA runtime None; recorded 2026-10-14T05:43:45+00:00
- Shapes: B=1, H=2, D=32, L in [32]; dtypes ['float32']; seed 0
- Timing: torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.2), timer=timeit.default_timer, num_threads=4; sample = mean per-call time of one block; 10 warmup calls per method; unit seconds
- Memory: unavailable on CPU
- SDPA backend: automatic selection; not verified
- Correctness: each method vs float64 reference; tolerances {'float32': {'atol': 1e-05, 'rtol': 1e-05}}

Reproduce (from the project root, inside the venv):

```bash
python -m attention_lab.benchmark --device cpu --preset smoke --output results/smoke.json
```

| L | dtype | causal | reference µs (IQR) | SDPA µs (IQR) | ref/SDPA | max abs err ref / SDPA | correct |
|---|---|---|---|---|---|---|---|
| 32 | float32 | noncausal | 24.4 (0.5) | 17.3 (0.4) | 1.41x | 2.6e-07 / 2.9e-07 | yes |
| 32 | float32 | causal | 39.1 (0.7) | 15.2 (0.5) | 2.58x | 3.6e-07 / 3.1e-07 | yes |

![latency vs L](smoke-latency.png)

### Observations (computed from this run)

- SDPA had the lower median latency in 2 of 2 configurations.
- float32, noncausal: reference/SDPA ratio 1.41x to 1.41x over L = 32.
- float32, causal: reference/SDPA ratio 2.58x to 2.58x over L = 32.

How to read this:
- One run on one machine. Re-run before quoting numbers; compare only rows from the same environment.
- Timings cover one attention call including the identical Python validation in both wrappers;
  input creation, correctness checks and I/O are excluded. This is not whole-model performance.
- SDPA backend: automatic selection; not verified. Nothing here shows which kernel ran.
- The score matrix is O(L²), but at small L fixed per-call costs (Python, op dispatch, allocation)
  are a large share, so latency can grow much more slowly than L². Growth near L² is only expected
  once the O(L²·D) arithmetic dominates.
- The explicit causal reference builds an [L, L] mask and runs masked_fill on every call, which adds
  work the noncausal path does not do.
