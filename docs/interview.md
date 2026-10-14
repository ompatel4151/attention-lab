# Interview prep

## Twelve questions, short answers

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
   Peak memory fell from 257 MiB to 9.2 MiB at L = 4096 (×8 growth for an 8× longer sequence, versus ×62
   for the reference). The cost is speed at small L: it's about 3.5× slower at L = 256, because a Python loop
   runs about 20 small ops per block. By L = 4096 it roughly matches the reference.

7. **Why is your tiled version still much slower than SDPA?**
   SDPA's flash kernel is fused C++. One call does all the blocks, with vectorized inner loops and no
   per-op dispatch or temporary tensors. Mine dispatches about 20 ops per block from Python. Same algorithm,
   very different implementation cost.

8. **Which SDPA backend ran, and how do you know?**
   The profiler showed `aten::_scaled_dot_product_flash_attention_for_cpu` for automatic SDPA on my M4.
   Forcing MATH with `sdpa_kernel` showed `_scaled_dot_product_attention_math`, which was no faster than my
   reference and used more memory. Forcing EFFICIENT_ATTENTION failed (no CPU kernel), and the benchmark
   recorded it as a skip.

9. **Why is SDPA's causal faster than its noncausal at large L, but not for your methods?**
   The flash kernel can skip key blocks that are entirely in the future. My reference and tiled versions
   compute every block and then mask. That's consistent with the measurements; I didn't read the kernel
   source to confirm it.

10. **How did you measure CPU memory, and how do you know it's right?**
    From torch.profiler's raw allocation and free records, summed in time order. The first version used
    per-op totals and under-reported the MATH kernel's peak. The number was impossible, because the MATH
    kernel must allocate L×L, and that exposed the bug. It was fixed with a regression test.

11. **What exactly did you time, and what pitfalls did you hit?**
    One forward call inside `inference_mode`, after warmup. `torch.utils.benchmark.Timer` defaults to 1
    thread, and its default clock syncs any accelerator (MPS on a Mac). I pass the real thread count and a
    plain clock. I report medians and IQRs and keep every raw sample.

12. **Did you write a GPU kernel? What doesn't this show?**
    No. Tiled is the algorithm in Python, and SDPA is PyTorch's kernel. Nothing here covers GPUs (that code
    is written but never run), training, or LLM performance. It's one attention call on a laptop CPU.

## Explain without reading (check each one off out loud)

- [ ] Write the formula and every tensor shape from `[B, H, L, D]` to the output.
- [ ] Draw the 4×4 causal mask and say which way True goes for `masked_fill` and for SDPA.
- [ ] Derive the online-softmax update and run the [0, ln 3 | ln 6] example on paper.
- [ ] Explain why the tiled causal case can't produce NaN when blocks run in order from key 0.
- [ ] State tiled's memory shapes and why the peak grows as L·(block + D).
- [ ] Explain why tiled training memory would be O(L²) again, and how FlashAttention avoids that.
- [ ] Explain atol/rtol and why exact equality is the wrong test.
- [ ] Draw the timing boundary and give both Timer pitfalls.
- [ ] Explain how the profiler gives both the kernel name and the peak memory, and the bug that was fixed.
- [ ] Explain why forced MATH is slow and flash is fast on the same input.
- [ ] Explain CUDA asynchrony and what `end.synchronize()` guarantees.
- [ ] Say what is *not* verified: everything CUDA, the fp16/bf16 tolerances, the reason flash causal is faster.

## Resume bullets

The numbers below come from `results/` (3 runs per preset, Apple M4 CPU, torch 2.14.1). Re-run before
using them, and only claim what you can explain from the checklist above.

**Attention Lab: attention correctness, memory and inference benchmark** (Python, PyTorch, pytest)
- Implemented the FlashAttention online-softmax recurrence in PyTorch. It matched the explicit reference
  within 7.6e-7 (float32 vs float64) and cut peak memory from 257 MiB to 9.2 MiB at sequence length 4096,
  with linear instead of quadratic growth, measured from profiler allocation records.
- Built a reproducible benchmark CLI (`torch.utils.benchmark`, inference mode, warmup, raw samples saved to
  JSON/CSV) with 77 pytest tests. Used the PyTorch profiler and `sdpa_kernel` to verify that PyTorch's SDPA
  dispatched its CPU flash kernel, which was 3.4–6.3× faster than the explicit implementation at L = 4096,
  while the forced math backend was not.
- Validated the memory measurement against known allocation sizes. A regression test covers a
  profiler-accounting bug that hid allocations freed inside a single op.

If you later run on an NVIDIA GPU, add a bullet:
- Measured peak GPU memory vs sequence length on [GPU]: reference [measured value] MiB vs SDPA
  [measured value] MiB at L=[value].
