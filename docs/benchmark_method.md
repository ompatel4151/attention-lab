# Benchmark method

## What is timed

The timed region is exactly one call to `method(q, k, v, causal=...)` inside `torch.inference_mode()`.
Every method runs the same `validate_qkv`, so its Python overhead is included equally for all of them.

These all happen **outside** the timed region:
- creating the inputs (a seeded CPU generator, then `.to(device, dtype)`), including host→device copies
- the correctness check against the float64 reference
- the memory probes
- JSON/CSV writing and plotting

## CPU timing

1. `--warmup` untimed calls (default 10) before the first round. Early calls pay one-time costs: lazy
   initialization, thread-pool startup, allocator growth and cold caches.
2. `torch.utils.benchmark.Timer(...).blocked_autorange(min_run_time=T / rounds)`. This picks a block size
   (calls per sample) so that each block is long enough for the clock to resolve, then keeps
   measuring blocks until the time is used up. Each sample is the mean per-call time of one block.
   T is 0.3 s by default and 1.5 s for `cpu-large`.
3. Samples from all rounds are pooled. We report the median, its bootstrap 95% CI, IQR, min and max in
   seconds, and keep every raw sample in the JSON.

Two pitfalls found in torch 2.14.1's `Timer` and handled explicitly:
- **`num_threads` defaults to 1.** We pass the real `torch.get_num_threads()` and record it. If you
  forget this, every CPU result is single-threaded. Drill: with 1 thread, L = 32 was *faster* and
  L = 256 much slower.
- **The default clock calls `torch.accelerator.synchronize()` on every read** when any accelerator
  exists. On Apple Silicon that is MPS. CPU-only timing has nothing to wait for, so we pass
  `timer=timeit.default_timer`.

## Interleaved rounds

Each configuration is timed in `--rounds` rounds (default 3). Every round times every method once, and
the starting method rotates each round. Without this, method A would always run before B, and slow
drift (thermal throttling, a background task, cache state) would bias one method systematically.
Interleaving spreads that drift across all methods. Separate runs of the same preset (`-run2`, `-run3`)
then measure between-run variation on top of that.

## Confidence intervals

- **Median:** a 95% percentile bootstrap. Resample the samples with replacement 1000 times, take each
  resample's median, and keep the 2.5th and 97.5th percentiles. It's seeded, so the result is reproducible.
- **Speedup** (reference median / method median): resample both sample lists independently, take the
  ratio each time, and report its percentiles. The report calls a method *faster* or *slower* only when
  this whole interval lies above or below 1.
- **Assumption and its limit:** the bootstrap treats samples as independent. Timing samples drift and
  correlate (thermal state, caches), so these intervals describe within-run noise and understate the real
  uncertainty. Between-run spread across `-run2`/`-run3` is the honest check, and it's often wider.

## GPU timing: MPS (run) and CUDA (written, not run)

GPU launches are asynchronous: `fn()` returns once the work is queued, not when it is finished.
Wall-clock timing without a synchronize would measure how fast the CPU queues work. `time_events`
serves both backends (`torch.mps` and `torch.cuda` have the same Event API):
1. Run the warmup calls, then a device-wide `synchronize()`, so warmup work can't spill into sample 1.
2. For each sample: `start.record()`, then `fn()`, then `end.record()`, then a device-wide `synchronize()`,
   then `start.elapsed_time(end)` (milliseconds, converted to seconds).
3. Events measure GPU stream time between the two markers. If the GPU sits idle waiting for the CPU to
   launch the next kernel, that idle time is included. This is exactly what hurts the tiled loop on MPS.

**An MPS hang found and worked around.** On torch 2.14.1, `end.synchronize()` followed by
`start.elapsed_time(end)` blocks forever on MPS, waiting inside `MPSEventPool::elapsedTime` (seen with
macOS `sample`, and reproduced in isolation in a subprocess with a timeout). A device-wide
`torch.mps.synchronize()` before `elapsed_time` works. The code uses the device-wide synchronize for
CUDA too: it's correct for both, just slightly more conservative.

## Memory on CPU

`profile_call` runs one call under `torch.profiler.profile(profile_memory=True)`. The profiler records
every allocation (+bytes) and free (−bytes) made through PyTorch's CPU allocator, each with a timestamp.
Summing those records in time order gives live bytes. The maximum is the peak above what was alive
before the call (q, k and v are not counted). Memory allocated outside PyTorch's allocator is invisible.

**A bug found and fixed while building this.** The first version summed per-op totals
(`self_cpu_memory_usage`) instead. A free that happens *inside* one op, such as SDPA's math kernel
releasing its [L, L] buffer, gets charged to the op as a whole and loses its timing. So that version
reported only the output size for `sdpa_math` (0.25 MiB instead of about 18 MiB at L = 1024), and it also
hid the flash kernel's working buffers. The fix reads the raw records (`prof.profiler.kineto_results`,
an undocumented attribute checked on torch 2.14.1). `test_peak_counts_allocations_freed_inside_one_op`
guards against a regression. Lesson: validate a measurement against a case where you know the answer.

## Which SDPA kernel ran

The same profiler pass lists every aten op the call dispatched. Ops starting with
`aten::_scaled_dot_product` are stored per method as `observed_sdpa_ops`. On the M4 CPU (torch 2.14.1):

| method | observed kernel |
|---|---|
| sdpa (automatic) | `aten::_scaled_dot_product_flash_attention_for_cpu` |
| sdpa_flash (`sdpa_kernel([SDPBackend.FLASH_ATTENTION])`) | same |
| sdpa_math (`sdpa_kernel([SDPBackend.MATH])`) | `aten::_scaled_dot_product_attention_math` |
| sdpa_efficient (`EFFICIENT_ATTENTION`) | none: skipped with "No viable backend ..." |

On **MPS** (same machine), automatic SDPA, forced MATH *and* forced FLASH all dispatch to
`aten::_scaled_dot_product_attention_math_for_mps`, a dedicated MPS kernel. Forcing a backend with
`sdpa_kernel` doesn't select a different implementation there. On MPS this "math" kernel is the fast path,
unlike on CPU.

This is an observation for these shapes and dtypes on this machine, not a rule. Forced methods also pay
the small cost of entering the `sdpa_kernel` context on every call.

## Memory (CUDA only)

For each method separately: `synchronize`, then `reset_peak_memory_stats()`, then
`baseline = memory_allocated()`, then one call, then `synchronize`, then `peak = max_memory_allocated()`.
- `baseline`: tensors already alive (q, k, v).
- `incremental_peak = peak − baseline`: the extra memory the call needed at its peak.
- This is **allocated** memory. The caching allocator *reserves* more than it allocates
  (`memory_reserved`), and that is not reported here.
- On CPU the profiler-based measurement above is used instead.

**Why the reference grows with L²:** it materializes `scores` and `weights`, each `[B, H, L, L]`,
plus a masked copy when causal. At B=1, H=4, L=1024 in fp32, each one is 4·1024²·4 B = 16 MiB, and
doubling L quadruples that. Fused backends can compute attention in tiles without storing the full
matrix (FlashAttention). Whether that happened in a given run has to be measured, not assumed.

## Correctness inside the benchmark

Each method's output is copied to the CPU and compared, outside timing, against `reference_attention`
run **on the CPU in float64** on the same already-cast inputs. MPS has no float64, and computing the
reference on the CPU keeps one reference for every device. Max |err| is recorded and checked against the
per-dtype tolerances in `benchmark.TOLERANCES`. If a check fails, the timings are kept but flagged, and
the CLI exits with code 1.

Measured on MPS (worst over all methods, L = 256…4096): float32 2.2e-6 (tolerance 1e-5), float16 2.3e-3
(tolerance 1e-2), bfloat16 1.8e-2 (tolerance 5e-2). That's at least 2.8× headroom. The largest errors came
from the reference and tiled methods, whose intermediates (scores, weights) are stored in the low-precision
dtype; SDPA's were lower (9.6e-4 / 7.6e-3), plausibly because its kernel accumulates in higher precision
(not verified). They're still unvalidated on CUDA, where different kernels may round differently.

## Skips and errors

- `--device cuda` without CUDA exits with a clear error. It never falls back.
- `--device auto` prints the device it picked.
- `cuda-sweep` on CPU is an error.
- bf16 that the GPU cannot run natively, out-of-memory, and unsupported dtype/operator combinations
  are recorded as `"status": "skipped"` with a reason.

## Pitfalls and limitations

- **Small L is dominated by overhead.** On the M4 run, latency grew about 3–4× while L grew 8×
  (32 → 256). Python, dispatch and allocation costs are a large share at that size.
- **The causal reference rebuilds its [L, L] mask on every call.** That is part of why causal
  reference is slower than noncausal. It is a property of this reference, not of attention itself.
- **SDPA's backend is chosen automatically.** The profiler records which kernel ran (see above). On CUDA, a
  CUDA tensor alone does not imply FlashAttention ran. Check `observed_sdpa_ops`.
- **Large L needs longer measurement.** `cpu-large` calls take up to about 40 ms, so that preset uses
  `min_run_time=1.5`. Even so, the same configuration's median varied by a median of 11% across 3 runs
  (up to 35%). That's wider than the within-run bootstrap CIs, so the README reports ranges across runs.
  For `cpu-sweep` the between-run spread was a median of 5%.
- **The tiled method is a Python loop.** Its latency includes about 20 aten op dispatches per key block.
  That's an implementation cost, not a property of the algorithm. On MPS every op is a GPU kernel launch,
  which is why tiled is 0.24–0.39× the reference's speed there at L = 4096. A fused kernel avoids it.
- **Noise:** other processes, thermal throttling and power mode all move results. On Apple Silicon,
  which cores run the work (performance or efficiency) can also change them. Treat ratios from one
  run as one observation.
- **Different environments are not comparable.** CPU and GPU runs measure different things and are
  reported in separate sections and plots.
- **Method order:** methods always run in the order given by `--methods`. Warmup reduces order effects
  but does not prove there are none.
- **FLOP/s is nominal.** The report divides 4·B·H·L²·D (the two matmuls, with causal counted as full) by
  the median latency. Softmax and masking aren't counted, and causal methods that skip blocks do less
  real work than this counts.
