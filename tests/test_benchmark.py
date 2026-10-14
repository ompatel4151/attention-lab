"""Benchmark: result arithmetic, output schema, skips and CPU memory. Never asserts which method is faster."""

import csv
import json

import pytest
import torch

from attention_lab import benchmark, reference_attention, sdpa_attention, tiled_attention
from attention_lab.benchmark import main, profile_call, speedup, summarize


def test_speedup_is_reference_over_candidate():
    assert speedup(2.0, 0.5) == 4.0  # candidate 4x faster
    assert speedup(1.0, 2.0) == 0.5  # candidate 2x slower
    with pytest.raises(ValueError):
        speedup(1.0, 0.0)


def test_summarize():
    s = summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    # statistics.quantiles default ("exclusive"): p25 = 1.5, p75 = 4.5 for 1..5.
    assert s == {"median_s": 3.0, "iqr_s": 3.0, "min_s": 1.0, "max_s": 5.0, "num_samples": 5}
    assert summarize([7.0])["iqr_s"] == 0.0


def test_smoke_run_schema(tmp_path):
    out = tmp_path / "run.json"
    assert main(["--device", "cpu", "--preset", "smoke", "--warmup", "2", "--min-run-time", "0.02", "--output", str(out)]) == 0

    data = json.loads(out.read_text())
    env, settings, results = data["environment"], data["settings"], data["results"]
    for key in ("python", "torch", "os", "torch_num_threads", "cuda_runtime", "gpu_name", "device"):
        assert key in env
    assert env["device"] == "cpu"
    assert settings["methods"] == ["reference", "sdpa", "tiled"]
    assert settings["memory_method"].startswith("torch.profiler")

    assert [(r["seq_len"], r["causal"]) for r in results] == [(32, False), (32, True)]
    for r in results:
        assert r["status"] == "ok"
        assert (r["batch"], r["heads"], r["head_dim"], r["dtype"]) == (1, 2, 32, "float32")
        for name, m in r["methods"].items():
            assert m["status"] == "ok" and m["correct"]
            t = m["timing"]
            assert t["num_samples"] == len(t["samples_s"]) > 0
            assert all(x > 0 for x in t["samples_s"])
            assert t["warmup_calls"] == 2
            assert m["memory"]["incremental_peak_bytes"] > 0
        ref = r["methods"]["reference"]["timing"]["median_s"]
        assert set(r["speedup_vs_reference"]) == {"sdpa", "tiled"}
        for name, ratio in r["speedup_vs_reference"].items():
            assert ratio == pytest.approx(ref / r["methods"][name]["timing"]["median_s"])
        # Only SDPA calls an SDPA kernel; the profiler sees which one.
        assert r["methods"]["sdpa"]["observed_sdpa_ops"]
        assert r["methods"]["reference"]["observed_sdpa_ops"] == r["methods"]["tiled"]["observed_sdpa_ops"] == []

    rows = list(csv.DictReader(out.with_suffix(".csv").open()))
    assert len(rows) == 6  # 2 causal flags x 3 methods
    assert {r["method"] for r in rows} == {"reference", "sdpa", "tiled"}


def test_failing_method_is_skipped_not_fatal(tmp_path, monkeypatch):
    def boom(q, k, v, causal=False):
        raise RuntimeError("No viable backend\nsecond line")

    monkeypatch.setitem(benchmark.METHODS, "boom", boom)
    out = tmp_path / "run.json"
    args = ["--methods", "reference", "boom", "--causal", "off", "--warmup", "1", "--min-run-time", "0.01"]
    assert main([*args, "--output", str(out)]) == 0
    (r,) = json.loads(out.read_text())["results"]
    assert r["methods"]["boom"] == {"status": "skipped", "skip_reason": "RuntimeError: No viable backend"}
    assert r["methods"]["reference"]["status"] == "ok"
    assert r["speedup_vs_reference"] == {}


def test_cpu_peak_memory_scaling():
    # The claim behind the tiled method: the reference holds [B, H, L, L] matrices (two
    # alive at once), so its peak grows with L^2; tiled holds [B, H, L, block] and
    # [B, H, L, D] temporaries, so its peak grows with L. Quadrupling L: ~16x vs ~4x.
    def peaks(L):
        q, k, v = (torch.randn(1, 2, L, 32) for _ in range(3))
        with torch.inference_mode():
            _, ref = profile_call(lambda: reference_attention(q, k, v, causal=True))
            _, tiled = profile_call(lambda: tiled_attention(q, k, v, causal=True, block_size=64))
        return ref, tiled

    (ref_s, tiled_s), (ref_l, tiled_l) = peaks(512), peaks(2048)
    assert ref_l >= 2 * (1 * 2 * 2048 * 2048 * 4)  # two float32 [1, 2, L, L] matrices
    assert ref_l / ref_s > 12
    assert tiled_l / tiled_s < 6
    assert tiled_l < ref_l / 8


def test_peak_counts_allocations_freed_inside_one_op():
    # Regression: SDPA's math backend allocates and frees an [L, L] buffer inside a single
    # aten op. An earlier per-op accounting reported only the output (~0.25 MiB here).
    L = 512
    q, k, v = (torch.randn(1, 2, L, 32) for _ in range(3))
    with torch.inference_mode():
        _, peak = profile_call(lambda: benchmark.METHODS["sdpa_math"](q, k, v))
    assert peak >= 1 * 2 * L * L * 4  # at least one float32 [1, 2, L, L] matrix


def test_profiler_sees_sdpa_kernel():
    q, k, v = (torch.randn(1, 2, 64, 32) for _ in range(3))
    with torch.inference_mode():
        ops, _ = profile_call(lambda: sdpa_attention(q, k, v, causal=True))
    assert any(op.startswith("aten::_scaled_dot_product") for op in ops)


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error path")
def test_explicit_cuda_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="torch.cuda.is_available"):
        main(["--device", "cuda", "--output", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):  # argparse error: CUDA preset on CPU is not silently substituted
        main(["--device", "cpu", "--preset", "cuda-sweep", "--output", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):  # no baseline -> no speedups -> refuse
        main(["--methods", "sdpa", "tiled", "--output", str(tmp_path / "x.json")])
