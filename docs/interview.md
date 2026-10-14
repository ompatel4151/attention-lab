# Interview prep

## Sixteen questions, short answers

1. **Why scale by 1/√D?**
   Each dot product sums D terms, so with unit-variance inputs its variance is about D. Scaling keeps
   the scores at about unit variance, so softmax doesn't saturate and gradients don't vanish.

2. **How does your causal mask work?**
   `ones(L, L).tril()` is True where key j ≤ query i. Blocked scores become −inf, so softmax gives them
   weight exactly 0. A test changes future tokens and checks that earlier outputs don't move, and also
   checks that they *do* move without the mask.

3. **What does True mean in SDPA's boolean mask? Why not pass a mask together with `is_causal=True`?**
   True means the position participates. `is_causal=True` already builds the lower-triangular mask, and
   PyTorch raises an error if both are set.

4. **Why don't your implementations match SDPA exactly, and how did you pick tolerances?**
   Float addition isn't associative, and the methods scale, block and reduce in different orders. I
   measured the errors first (≤ 8.9e-7 against a float64 reference across every run) and set the
   tolerances with about 10× headroom. A real bug (scaling by 1/D) gives an error near 0.9.

5. **Explain online softmax. Why is the tiled result exact?**
   For each row, keep a running max m, a sum l = Σ exp(s − m) and acc = Σ exp(s − m)·v. When a block raises
   the max, multiply l and acc by exp(m_old − m_new) to move them to the new max. At the end, acc / l is
   exactly softmax-weighted V. Example: scores [0, ln 3 | ln 6] with v = [2, 4 | 6] give 5 either way.

6. **What did tiling buy you, and what did it cost?**
   Memory: at L = 4096 the peak fell from 257 MiB to 8.2 MiB, growing ×8 for an 8× longer sequence
   versus ×62 for the reference. Speed: at small L a Python loop of about 20 ops per block is slow (0.29×
   the reference's speed at L = 512). At L = 4096 it's at parity for noncausal and 1.7–1.9× *faster* for
   causal, across three runs.

7. **How did you make causal faster, and how did you find what to fix?**
   For the key block at `start`, rows before `start` can't see it, so only rows `start…L−1` are processed.
   That's about half the work. The first version didn't get faster, and a profile showed why: building a
   full [L × block] mask, `masked_fill` and a `torch.cat` every block ate the savings. Masking only the
   [block × block] diagonal tile, and emitting rows as soon as they've seen their last key, fixed it.
   A test counts the rows fed to each block (10, 6, 2 for L = 10, block 4).

8. **Which SDPA backend ran, and how do you know?**
   The profiler showed `aten::_scaled_dot_product_flash_attention_for_cpu` for automatic SDPA on my M4.
   Forcing MATH with `sdpa_kernel` showed `_scaled_dot_product_attention_math`, which was no faster than my
   reference and used more memory. Forcing EFFICIENT_ATTENTION failed (no CPU kernel), and the benchmark
   recorded it as a skip.

9. **Why is SDPA's causal faster than its noncausal at large L?**
   Like my tiled version, the flash kernel can skip key blocks that are entirely in the future. The
   reference and forced MATH compute every block and then mask, so their causal runs are slower than
   noncausal. That's consistent with the data; I didn't read the kernel source to confirm it.

10. **How did you measure CPU memory, and how do you know it's right?**
    From torch.profiler's raw allocation and free records, summed in time order. The first version used
    per-op totals and under-reported the MATH kernel's peak. The number was impossible, because the MATH
    kernel must allocate L×L, and that exposed the bug. It was fixed with a regression test.

11. **What exactly did you time, and what pitfalls did you hit?**
    One forward call inside `inference_mode`, after warmup. `torch.utils.benchmark.Timer` defaults to 1
    thread, and its default clock syncs any accelerator (MPS on a Mac). I pass the real thread count and a
    plain clock. I report medians and IQRs and keep every raw sample.

12. **Did you write a GPU kernel? What doesn't this show?**
    No. Tiled is the algorithm in Python, `torch.compile` generated the fused code for the compiled
    variant, and SDPA is PyTorch's kernel. Nothing here covers CUDA (that code is written but never run),
    training, or LLM performance.

13. **Did torch.compile help?**
    Barely, and the details are the interesting part. Compiling the whole function unrolled the Python loop
    and ran for over 10 minutes at L = 1024, so I compiled only the per-block step: one graph, no graph
    breaks. With static shapes it was about 10% faster than eager; forced dynamic shapes were slower. Across
    three runs at L = 4096 its speedup wasn't statistically clear (intervals include 1). It did halve tiled's
    noncausal memory (4.2 vs 8.2 MiB), because fused ops don't materialise intermediate tensors.

14. **How did you make the timing statistically defensible?**
    Methods run interleaved in rounds with a rotating start, so drift doesn't always hit the same method.
    I report medians with 95% bootstrap CIs and call something faster only if the whole ratio CI is above 1.
    Then I repeated each preset three times. The between-run spread (median 11%, up to 35% at large L) was
    wider than the within-run CIs, which is exactly the bootstrap's known blind spot: it assumes independent
    samples. So the README quotes ranges across runs.

15. **What happened on the Apple GPU (MPS)?**
    The tiled loop lost badly: 0.24–0.39× the reference's speed at L = 4096, because every small op is a
    separate GPU kernel launch. SDPA was 2.1–8.1× faster than the reference at L = 4096, and it dispatches
    to a dedicated MPS kernel. MPS has no peak-memory counter,
    so memory is reported as unavailable, not estimated. I also hit a PyTorch hang:
    `end.synchronize()` then `elapsed_time()` on MPS events never returns. I located it with macOS `sample`,
    reproduced it in isolation, and worked around it with a device-wide synchronize.

16. **How do you know fp16/bf16 results are acceptable?**
    Every method is compared with a float64 reference computed on the CPU from the same already-rounded
    inputs, so the error measured is computation error only. On MPS the worst errors were 2.3e-3 (fp16,
    tolerance 1e-2) and 1.8e-2 (bf16, tolerance 5e-2), at least 2.8× headroom.
    They're still unvalidated on CUDA, where kernels may round differently.

## Explain without reading (check each one off out loud)

- [ ] Write the formula and every tensor shape from `[B, H, L, D]` to the output.
- [ ] Draw the 4×4 causal mask and say which way True goes for `masked_fill` and for SDPA.
- [ ] Derive the online-softmax update and run the [0, ln 3 | ln 6] example on paper.
- [ ] Explain the causal row skipping: which rows each key block sees, why only the diagonal tile needs a
      mask, and why rows can be emitted early.
- [ ] Explain why the tiled causal case can never produce NaN.
- [ ] State tiled's memory shapes and why the peak grows as L·(block + D).
- [ ] Explain why compiling the whole loop failed and compiling the step worked, and static vs dynamic shapes.
- [ ] Explain atol/rtol, why exact equality is the wrong test, and why the golden reference runs on the CPU.
- [ ] Draw the timing boundary, give both Timer pitfalls, and explain interleaved rounds.
- [ ] Explain a bootstrap CI and why the between-run spread is wider than it.
- [ ] Explain how the profiler gives both the kernel name and the peak memory, and the bug that was fixed.
- [ ] Explain GPU asynchrony, what synchronize guarantees, and the MPS event hang.
- [ ] Say what is *not* verified: everything CUDA, the reason flash causal is faster.

## Resume bullets

The numbers below come from `results/` (Apple M4, torch 2.14.1; CPU presets run 3×, MPS once). Re-run before
using them, and only claim what you can explain from the checklist above.

**Attention Lab: attention correctness, memory and inference benchmark** (Python, PyTorch, pytest)
- Implemented FlashAttention-style online-softmax attention in PyTorch with causal block skipping. It
  matched a float64 reference within 7.6e-7, cut peak memory 31× (257 → 8.2 MiB) at sequence length 4096,
  and ran causal attention 1.7–1.9× faster than the explicit implementation across three runs.
- Profiled a causal version that failed to speed up, traced the loss to per-block masking and concatenation
  overhead, and fixed it with diagonal-tile masking and early row emission.
- Built a reproducible CPU/Apple-GPU benchmark (interleaved rounds, bootstrap 95% CIs, raw samples in
  JSON/CSV, 82 pytest tests). Used the PyTorch profiler and `sdpa_kernel` to show SDPA's CPU speedup
  (3.2–6.4× at L = 4096) comes from its flash kernel, and validated fp16/bf16 error bounds on MPS.

If you later run on an NVIDIA GPU, add a bullet:
- Measured peak GPU memory vs sequence length on [GPU]: reference [measured value] MiB vs SDPA
  [measured value] MiB at L=[value].
