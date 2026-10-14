"""Milestone 1 correctness tests (CPU, float32).

Why not exact equality between reference and SDPA?
Float addition is not associative: (a + b) + c can differ from a + (b + c) in
the last bits. SDPA may scale q before the matmul instead of after, block the
computation, fuse softmax, or use different SIMD reduction orders. Same math,
different rounding. Measured on this machine (torch 2.14.1, Apple M4, shapes up
to [1, 2, 256, 32]): forward max |err| <= 6e-7, gradient max |err| <= 2.2e-6,
i.e. a handful of float32 ULPs (eps = 1.19e-7). The tolerances below leave
~10x headroom over that; a real bug (wrong scale, missing mask) gives errors
of order 1e-2 or larger.

torch.testing.assert_close checks |actual - expected| <= atol + rtol * |expected|
elementwise, and on failure prints the greatest absolute and relative
difference and where it occurred.
"""

import pytest
import torch

from attention_lab import causal_mask, reference_attention, sdpa_attention, tiled_attention

FWD_TOL = dict(atol=1e-5, rtol=1e-5)
GRAD_TOL = dict(atol=5e-5, rtol=1e-4)
ALL_METHODS = [reference_attention, sdpa_attention, tiled_attention]


def make_qkv(shape, seed=0):
    # A local Generator keeps tests deterministic without touching global RNG state.
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape, generator=g) for _ in range(3))


def test_causal_mask_layout():
    expected = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    assert torch.equal(causal_mask(3), expected)


@pytest.mark.parametrize("fn", ALL_METHODS)
def test_hand_checkable_example(fn):
    # B=H=1, L=2, D=1 so the scale 1/sqrt(D) is 1.
    # q0 = ln 3, k = [1, 0] -> row-0 scores [ln 3, 0] -> softmax [3/4, 1/4].
    # q1 = 0 -> row-1 scores [0, 0] -> softmax [1/2, 1/2].
    # v = [2, 4].
    q = torch.tensor([torch.log(torch.tensor(3.0)), 0.0]).view(1, 1, 2, 1)
    k = torch.tensor([1.0, 0.0]).view(1, 1, 2, 1)
    v = torch.tensor([2.0, 4.0]).view(1, 1, 2, 1)

    # Noncausal: row0 = 3/4*2 + 1/4*4 = 2.5, row1 = (2+4)/2 = 3.
    torch.testing.assert_close(fn(q, k, v), torch.tensor([2.5, 3.0]).view(1, 1, 2, 1))
    # Causal: row0 sees only key 0 -> 2.0; row1 unchanged -> 3.0.
    torch.testing.assert_close(fn(q, k, v, causal=True), torch.tensor([2.0, 3.0]).view(1, 1, 2, 1))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 32, 32), (2, 3, 17, 8), (1, 1, 1, 4)])
def test_reference_matches_sdpa(shape, causal):
    q, k, v = make_qkv(shape)
    ref = reference_attention(q, k, v, causal=causal)
    out = sdpa_attention(q, k, v, causal=causal)
    assert ref.shape == out.shape == shape
    torch.testing.assert_close(out, ref, **FWD_TOL)


@pytest.mark.parametrize("fn", ALL_METHODS)
def test_causal_ignores_future_tokens(fn):
    L, t = 8, 5
    q, k, v = make_qkv((1, 2, L, 4))
    # Change every position >= t in q, k and v.
    q2, k2, v2 = (x.clone() for x in (q, k, v))
    for x in (q2, k2, v2):
        x[..., t:, :] += 10.0

    before = fn(q, k, v, causal=True)
    after = fn(q2, k2, v2, causal=True)
    torch.testing.assert_close(after[..., :t, :], before[..., :t, :], **FWD_TOL)
    # Sanity: the perturbation is big enough to matter where it is visible...
    assert not torch.allclose(after[..., t:, :], before[..., t:, :])
    # ...and without the mask, earlier rows DO change. So this test would catch a missing mask.
    assert not torch.allclose(fn(q2, k2, v2)[..., :t, :], fn(q, k, v)[..., :t, :])


_good = torch.randn(1, 1, 4, 8)


@pytest.mark.parametrize("fn", ALL_METHODS)
@pytest.mark.parametrize(
    "q, k, v, err",
    [
        (torch.randn(1, 4, 8), _good, _good, ValueError),  # 3-D, missing heads dim
        (_good, torch.randn(1, 1, 5, 8), _good, ValueError),  # sequence length mismatch
        (_good, _good, torch.randn(1, 1, 4, 16), ValueError),  # head_dim mismatch
        (torch.randn(1, 1, 0, 8),) * 3 + (ValueError,),  # empty
        (_good, _good.double(), _good, TypeError),  # dtype mismatch
        (_good.long(),) * 3 + (TypeError,),  # integer dtype
        (_good, _good.to("meta"), _good, ValueError),  # device mismatch (meta = shape-only device, always available)
        (_good.numpy(), _good, _good, TypeError),  # not a tensor
    ],
)
def test_invalid_inputs_raise(fn, q, k, v, err):
    with pytest.raises(err):
        fn(q, k, v)


@pytest.mark.parametrize("causal", [False, True])
def test_gradients_match_sdpa(causal):
    shape = (2, 2, 16, 8)
    q, k, v = make_qkv(shape)
    # Stand-in for dLoss/dOutput arriving from later layers.
    grad_out = torch.randn(shape, generator=torch.Generator().manual_seed(1))

    def grads(fn):
        # Fresh leaf tensors per method so gradients don't accumulate across runs.
        inputs = [x.clone().requires_grad_() for x in (q, k, v)]
        out = fn(*inputs, causal=causal)
        # Vector-Jacobian product: returns (dL/dq, dL/dk, dL/dv) for L = sum(out * grad_out).
        return torch.autograd.grad(out, inputs, grad_outputs=grad_out)

    g_sdpa = grads(sdpa_attention)
    for method in (reference_attention, tiled_attention):
        for name, g, g_s in zip("qkv", grads(method), g_sdpa):
            torch.testing.assert_close(g, g_s, **GRAD_TOL, msg=lambda m: f"{method.__name__} d{name}: {m}")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [1, 7, "L", 1000])  # 7 does not divide L; "L" = one block; 1000 > L
@pytest.mark.parametrize("shape", [(1, 2, 32, 32), (2, 3, 17, 8), (1, 1, 1, 4)])
def test_tiled_matches_reference(shape, block_size, causal):
    q, k, v = make_qkv(shape)
    bs = shape[2] if block_size == "L" else block_size
    out = tiled_attention(q, k, v, causal=causal, block_size=bs)
    assert out.shape == shape
    torch.testing.assert_close(out, reference_attention(q, k, v, causal=causal), **FWD_TOL)


def test_tiled_rescales_when_max_rises():
    # One row, scores [0, ln 3 | ln 6] with block_size=2: block 2 raises the running max,
    # so the old sums must be rescaled by exp(ln 3 - ln 6) = 1/2. Exact answer:
    # softmax weights [1, 3, 6] / 10 over v = [2, 4, 6] -> 5.0.
    # D=1 and q=1, so row 0's scores are exactly k. Only row 0 is checked.
    q = torch.ones(1, 1, 3, 1)
    k = torch.log(torch.tensor([1.0, 3.0, 6.0])).view(1, 1, 3, 1)
    v = torch.tensor([2.0, 4.0, 6.0]).view(1, 1, 3, 1)
    torch.testing.assert_close(tiled_attention(q, k, v, block_size=2)[0, 0, 0], torch.tensor([5.0]))


@pytest.mark.parametrize("block_size", [0, -1, 2.5])
def test_tiled_rejects_bad_block_size(block_size):
    q, k, v = make_qkv((1, 1, 4, 8))
    with pytest.raises(ValueError):
        tiled_attention(q, k, v, block_size=block_size)
