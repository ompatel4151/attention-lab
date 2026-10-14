"""Scaled dot-product attention: a readable reference and a thin SDPA wrapper.

Shapes, everywhere in this module:
    q, k, v : [B, H, L, D]   batch, heads, sequence length, head dim
    scores  : [B, H, L, L]   one score per (query position, key position)
    output  : [B, H, L, D]

Milestone 1 scope: self-attention only (q, k, v share one shape), no GQA,
no padding mask, no dropout, no KV cache.
"""

import math

import torch
import torch.nn.functional as F


def validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Raise a clear error unless q, k, v are non-empty 4-D floating tensors
    with identical shape, dtype and device."""
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D [B, H, L, D], got shape {tuple(t.shape)}")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"q, k, v must have the same shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.numel() == 0:
        raise ValueError(f"q, k, v must be non-empty, got shape {tuple(q.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if not q.dtype.is_floating_point:
        raise TypeError(f"q, k, v must be floating point, got {q.dtype}")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q, k, v must be on one device, got {q.device}, {k.device}, {v.device}")


def causal_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Boolean [L, L] mask; True = this (query row, key column) pair may attend.

    Row i is query position i, column j is key position j. Allowed iff j <= i,
    i.e. a token sees itself and the past, never the future. tril() keeps the
    lower triangle including the diagonal and zeroes the rest.
    Same convention as SDPA's boolean attn_mask: True participates.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """softmax((q @ k^T) / sqrt(D) + mask) @ v, written out step by step."""
    validate_qkv(q, k, v)
    d = q.size(-1)

    # k.transpose(-2, -1): [B, H, L, D] -> [B, H, D, L] (swaps last two dims, no copy).
    # @ on 4-D tensors is a batched matmul over the leading [B, H] dims:
    # [B, H, L, D] @ [B, H, D, L] -> [B, H, L, L].
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

    if causal:
        # Adding an additive mask of 0 (allowed) / -inf (blocked) is the same as
        # overwriting blocked scores with -inf. masked_fill does the latter;
        # the [L, L] mask broadcasts across B and H. exp(-inf) == 0, so blocked
        # keys get exactly zero weight. Every row keeps its diagonal, so no row
        # is all -inf (which would give 0/0 = NaN).
        scores = scores.masked_fill(~causal_mask(q.size(-2), q.device), float("-inf"))

    # torch.softmax subtracts the row max before exp internally, so large
    # scores don't overflow. dim=-1 normalizes over keys: each row sums to 1.
    weights = torch.softmax(scores, dim=-1)

    # [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]: each output row is a
    # weighted average of value rows.
    return weights @ v


def sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Same semantics as reference_attention, using PyTorch's fused operator.

    torch.nn.functional.scaled_dot_product_attention picks a backend (math,
    flash, memory-efficient, ...) at runtime; which one ran is not reported.
    - dropout_p=0.0 explicitly: SDPA applies dropout whenever dropout_p > 0,
      even in eval mode, so we never rely on the default.
    - is_causal=True makes SDPA build the same lower-triangular mask internally.
      We pass attn_mask=None with it: giving both is an error / ambiguous.
    - If you ever pass a boolean attn_mask yourself: True means that position
      DOES participate (the opposite of masked_fill's "True = fill").
    Docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    validate_qkv(q, k, v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
