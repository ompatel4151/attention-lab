# Design

## Architecture

There are three plain modules and no classes or framework:

```
attention.py ──► benchmark.py ──► results/*.json, *.csv ──► report.py ──► reports/report.md, *.png
     ▲
tests/test_attention.py
```

- `attention.py` holds the math: `validate_qkv`, `causal_mask`, `reference_attention`, and `sdpa_attention`.
- `benchmark.py` imports both attention functions and treats them as black boxes with one signature:
  `fn(q, k, v, causal=bool) -> Tensor`.
- `report.py` reads only JSON files. It never calls the attention code, so it cannot make up numbers.

## Tensor shapes

Symbols: B = batch, H = heads, L = sequence length, D = head dim. Every head runs independently.

| step | expression | shape |
|---|---|---|
| inputs | `q, k, v` | `[B, H, L, D]` |
| transpose keys | `k.transpose(-2, -1)` | `[B, H, D, L]` (a view, no copy) |
| scores | `q @ kᵀ / √D` | `[B, H, L, L]` |
| mask | `masked_fill(~causal_mask(L), -inf)` | `[B, H, L, L]`; the mask `[L, L]` broadcasts |
| weights | `softmax(scores, dim=-1)` | `[B, H, L, L]`; every row sums to 1 |
| output | `weights @ v` | `[B, H, L, D]` |

`@` on 4-D tensors is a batched matrix multiply: it multiplies the last two dims and treats the
leading `[B, H]` dims as a batch.

**Why √D?** If the entries of q and k have variance about 1, each dot product sums D terms, so its
variance is about D. Dividing by √D brings the variance back to about 1. Without that, softmax
saturates toward one-hot and its gradients vanish.

## Masking

- `causal_mask(L) = ones(L, L).tril()`: True on and below the diagonal. Row i is query i and column j
  is key j, and position (i, j) is allowed iff j ≤ i.
- **True means "may attend"**. SDPA's boolean `attn_mask` uses the same convention. Watch out:
  `masked_fill(mask, x)` fills where the mask is True, so we pass it `~allowed`.
- Overwriting blocked scores with −inf is the same as adding a 0/−inf mask. Since exp(−inf) = 0,
  future keys get exactly zero weight.
- Every causal row keeps its diagonal, so no row is all −inf. An all −inf row would give
  softmax 0/0 = NaN. Padding masks (not implemented) would need to handle that case.
- `sdpa_attention` uses `is_causal=True` and `attn_mask=None`. It never passes both.
- **Not handled yet:** rectangular attention, where the query length differs from the key length
  (as with a KV cache). Causal alignment is ambiguous there (top-left vs bottom-right), so it was
  left out deliberately.

The test `test_causal_ignores_future_tokens` changes q, k and v at positions ≥ t and asserts that
outputs before t stay the same. It also asserts that *without* the mask they change, which shows the
test would catch a missing mask.

## Autograd vs inference

- **Training/autograd:** PyTorch records each operation in a graph so that `backward()` (or
  `torch.autograd.grad`) can apply the chain rule. `test_gradients_match_sdpa` passes the same
  upstream gradient into both implementations and compares dq, dk and dv. That checks that SDPA's
  backward matches the gradient autograd derives from our explicit operations.
- **Inference:** `torch.inference_mode()` turns graph recording off completely, which saves memory
  and time. Tensors created inside it can never be used in backward. Every benchmark runs inside
  it, so the timings are forward-only.

## Numerical error

- Floating-point addition is not associative. SDPA may scale before the matmul, split the work into
  blocks, or reduce in a different order, so its results differ from ours in the last bits.
- The tests use float32 tolerances `atol=rtol=1e-5` for the forward pass and `atol=5e-5, rtol=1e-4`
  for gradients. Before choosing them we measured max |err| ≤ 6e-7 (forward) and ≤ 2.2e-6
  (gradient), which is a few float32 ULPs (eps ≈ 1.19e-7).
- The benchmark compares *both* methods against a float64 reference built from the same
  (already-cast) inputs, so the reference itself has negligible error. fp16/bf16 tolerances are
  looser (eps 9.8e-4 and 7.8e-3) and **have not been validated on a GPU**.
- `torch.softmax` subtracts the row maximum before exponentiating, so large scores do not overflow.
  A hand-written `exp(x) / exp(x).sum()` can overflow to inf/NaN.

## Validation

`validate_qkv` rejects the following: non-tensors, anything that isn't 4-D, unequal shapes (no GQA),
empty tensors, mixed dtypes, non-float dtypes, and mixed devices. It raises `ValueError` for
shape/device problems and `TypeError` for type problems.

## Limitations

There is no GQA, padding mask, dropout, KV cache, rectangular causal attention, or MPS/other
accelerator support. The benchmark measures one isolated attention call, not a model.
