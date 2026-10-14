"""Milestone 2: result arithmetic and output schema. Never asserts which method is faster."""

import csv
import json

import pytest
import torch

from attention_lab.benchmark import main, speedup, summarize


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
    assert settings["memory_method"] == "unavailable on CPU"
    assert settings["sdpa_backend"] == "automatic selection; not verified"

    assert [(r["seq_len"], r["causal"]) for r in results] == [(32, False), (32, True)]
    for r in results:
        assert r["status"] == "ok" and r["correctness"]["passed"]
        assert (r["batch"], r["heads"], r["head_dim"], r["dtype"]) == (1, 2, 32, "float32")
        medians = {}
        for name in ("reference", "sdpa"):
            t = r["methods"][name]["timing"]
            assert t["num_samples"] == len(t["samples_s"]) > 0
            assert all(x > 0 for x in t["samples_s"])
            assert t["warmup_calls"] == 2
            assert r["methods"][name]["memory"] is None
            medians[name] = t["median_s"]
        assert r["speedup_reference_over_sdpa"] == pytest.approx(medians["reference"] / medians["sdpa"])

    rows = list(csv.DictReader(out.with_suffix(".csv").open()))
    assert len(rows) == 4  # 2 causal flags x 2 methods
    assert {r["method"] for r in rows} == {"reference", "sdpa"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error path")
def test_explicit_cuda_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="torch.cuda.is_available"):
        main(["--device", "cuda", "--output", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):  # argparse error: CUDA preset on CPU is not silently substituted
        main(["--device", "cpu", "--preset", "cuda-sweep", "--output", str(tmp_path / "x.json")])
