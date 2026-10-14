# Interview prep

## Eight questions, short answers

1. **Why scale by 1/√D?**
   Each dot product sums D terms, so with unit-variance inputs its variance is about D. Scaling
   keeps the scores at about unit variance, so softmax doesn't saturate and gradients don't vanish.

2. **How does your causal mask work?**
   `ones(L, L).tril()` is True where key j ≤ query i. Blocked scores become −inf, so softmax gives
   them weight exactly 0. My test changes future tokens and checks that earlier outputs don't move,
   and also checks that they *do* move without the mask.

3. **What does True mean in SDPA's boolean mask? Why not pass a mask together with `is_causal=True`?**
   True means the position participates. `is_causal=True` already builds the lower-triangular mask
   internally, and passing both is disallowed or ambiguous.

4. **Why don't your reference and SDPA match exactly, and how did you pick tolerances?**
   Float addition isn't associative, and SDPA scales, blocks and reduces in a different order. I
   measured the actual max error first (≤ 6e-7 forward, ≤ 2.2e-6 for gradients in float32) and set
   tolerances with about 10× headroom. A real bug, like scaling by 1/D, gives an error near 0.9.

5. **Why compare gradients if the benchmark is inference-only?**
   To learn and verify autograd: SDPA's backward must agree with the gradient autograd derives from
   my explicit operations. The benchmarks use `inference_mode()`, so no graph is recorded and the
   timings are forward-only.

6. **Why warm up, and why synchronize on CUDA?**
   Warmup moves one-time costs out of the samples (lazy init, allocation, cold caches). CUDA calls
   return before the GPU finishes, so without events or a synchronize you time how fast work is
   queued, not how long it runs.

7. **Why does the reference need O(L²) memory, and how can SDPA avoid that?**
   It stores `[B, H, L, L]` scores and weights. Tiled kernels (FlashAttention) keep a running max
   and sum and process key blocks one at a time, so they never store the full matrix.

8. **Did you write a GPU kernel? What does your speedup show?**
   No. I called PyTorch's fused operator, so I'm using PyTorch's kernels. My result shows that on
   [hardware], at these small shapes, SDPA had a lower median than my explicit reference
   ([measured value]). It doesn't show which backend ran, and it doesn't say anything about
   whole-model performance.

## Explain without reading (check each one off out loud)

- [ ] Write the formula and every tensor shape from `[B, H, L, D]` to the output.
- [ ] Draw the 4×4 causal mask and say which way True goes for `masked_fill` and for SDPA.
- [ ] Explain why a causal row can never be all −inf, and when a padding mask could make one.
- [ ] Explain `transpose(-2, -1)` and batched `@`.
- [ ] Explain atol/rtol and why exact equality is the wrong test.
- [ ] Explain what `torch.autograd.grad(out, inputs, grad_outputs=g)` computes.
- [ ] Draw the timing boundary: what is inside the timed region and what is outside.
- [ ] Give the two `Timer` pitfalls (num_threads=1, the accelerator-sync clock) and why they matter.
- [ ] Explain CUDA asynchrony and what `end.synchronize()` guarantees.
- [ ] Explain baseline vs incremental peak vs reserved memory.
- [ ] Explain why latency grew less than L² on CPU at L ≤ 256.
- [ ] Say what is *not* verified: the SDPA backend, everything on CUDA, and the fp16/bf16 tolerances.

## Resume bullets

The numbers below come from `results/cpu-sweep*.json` (3 runs, Apple M4 CPU, torch 2.14.1). Re-run
before you use them, and only claim what you can explain from the checklist above.

**Attention Lab: PyTorch attention correctness and inference benchmark** (Python, PyTorch, pytest)
- Implemented scaled dot-product attention from tensor ops with an explicit causal mask. Validated
  it against PyTorch's fused SDPA with 35 pytest tests covering hand-computed cases, future-token
  perturbation, invalid inputs and autograd gradients (max float32 error 6.7e-7 vs float64).
- Built a reproducible CLI benchmark (`torch.utils.benchmark`, inference mode, seeded inputs, warmup,
  raw samples saved to JSON/CSV). Accounted for two timing pitfalls: Timer's single-thread
  default and an accelerator sync in its default clock.
- Measured fused SDPA at 1.9–2.7× lower median latency than the explicit implementation at L=256 on
  an Apple M4 CPU, consistent across 3 runs. Analyzed why latency grew sub-quadratically at small L.

If you later run on an NVIDIA GPU, a fourth bullet becomes possible:
- Measured peak GPU memory vs sequence length on [GPU]: explicit reference [measured value] MiB vs
  SDPA [measured value] MiB at L=[value].

Replace or add one bullet describing the change you made yourself.

Don't use the "custom kernel" or "optimized attention" wording. You used and measured an existing operator.
