# Benchmark method

## What is timed

The timed region is exactly one call to `method(q, k, v, causal=...)` inside `torch.inference_mode()`.
Both wrappers run the same `validate_qkv`, so its Python overhead is included equally on both sides.

These all happen **outside** the timed region:
- creating the inputs (a seeded CPU generator, then `.to(device, dtype)`), including host→device copies
- the correctness check against the float64 reference
- the memory probes
- JSON/CSV writing and plotting

## CPU timing

1. `--warmup` untimed calls (default 10). Early calls pay one-time costs: lazy initialization,
   thread-pool startup, allocator growth and cold caches.
2. `torch.utils.benchmark.Timer(...).blocked_autorange(min_run_time=0.2)`. This picks a block size
   (calls per sample) so that each block is long enough for the clock to resolve, then keeps
   measuring blocks until `min_run_time` has passed. Each sample is the mean per-call time of one block.
3. We report median and IQR (p75 − p25) in seconds, and keep every raw sample in the JSON.

Two pitfalls found in torch 2.14.1's `Timer` and handled explicitly:
- **`num_threads` defaults to 1.** We pass the real `torch.get_num_threads()` and record it. If you
  forget this, every CPU result is single-threaded.
- **The default clock calls `torch.accelerator.synchronize()` on every read** when any accelerator
  exists. On Apple Silicon that is MPS. CPU-only timing has nothing to wait for, so we pass
  `timer=timeit.default_timer`.

## CUDA timing (implemented, not run)

CUDA kernel launches are asynchronous: `fn()` returns once the work is queued, not when it is
finished. Wall-clock timing without a synchronize would measure how fast the CPU queues work.
The approach:
1. Run the warmup calls, then `torch.cuda.synchronize()` so warmup work cannot spill into sample 1.
2. For each sample: `start.record()`, then `fn()`, then `end.record()`, then `end.synchronize()`,
   then `start.elapsed_time(end)` (milliseconds, converted to seconds).
3. Events measure GPU stream time between the two markers. If the GPU sits idle waiting for the CPU
   to launch the next kernel, that idle time is included. This matters for the multi-kernel
   reference at small L.

## Memory (CUDA only)

For each method separately: `synchronize`, then `reset_peak_memory_stats()`, then
`baseline = memory_allocated()`, then one call, then `synchronize`, then `peak = max_memory_allocated()`.
- `baseline`: tensors already alive (q, k, v).
- `incremental_peak = peak − baseline`: the extra memory the call needed at its peak.
- This is **allocated** memory. The caching allocator *reserves* more than it allocates
  (`memory_reserved`), and that is not reported here.
- CPU memory is recorded as unavailable. It is not estimated.

**Why the reference grows with L²:** it materializes `scores` and `weights`, each `[B, H, L, L]`,
plus a masked copy when causal. At B=1, H=4, L=1024 in fp32, each one is 4·1024²·4 B = 16 MiB, and
doubling L quadruples that. Fused backends can compute attention in tiles without storing the full
matrix (FlashAttention). Whether that happened in a given run has to be measured, not assumed.

## Correctness inside the benchmark

Each method's output is compared, outside timing, against `reference_attention` on float64 copies of
the same inputs, using the per-dtype tolerances in `benchmark.TOLERANCES`. Max |err| is recorded.
If a check fails, the timings are kept but flagged, and the CLI exits with code 1.

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
- **SDPA's backend is chosen automatically and not verified.** On CUDA, a CUDA tensor does not imply
  FlashAttention ran. `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` and similar can force a
  backend (available in 2.14.1) but that is not used yet. A profiler trace would show the kernels.
- **Noise:** other processes, thermal throttling and power mode all move results. On Apple Silicon,
  which cores run the work (performance or efficiency) can also change them. Treat ratios from one
  run as one observation.
- **Different environments are not comparable.** CPU and GPU runs measure different things and are
  reported in separate sections and plots.
- **Method order:** the reference always runs before SDPA. Warmup reduces order effects but does
  not prove there are none.
