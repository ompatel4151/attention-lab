"""Milestone 3: the report is built only from real saved runs."""

import pytest
import torch

from attention_lab import benchmark, report


def test_no_results_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty dir: no results/*.json
    report.main(["--out-dir", "rep"])
    text = (tmp_path / "rep" / "report.md").read_text()
    assert "No benchmark results found" in text
    assert "µs" not in text
    assert not list((tmp_path / "rep").glob("*.png"))


def test_report_from_real_smoke_run(tmp_path):
    run = tmp_path / "smoke.json"
    benchmark.main(["--preset", "smoke", "--warmup", "2", "--min-run-time", "0.02", "--output", str(run)])
    report.main([str(run), "--out-dir", str(tmp_path / "rep")])

    text = (tmp_path / "rep" / "report.md").read_text()
    assert "automatic selection" in text
    for method in ("reference", "sdpa", "tiled"):
        assert f"| 32 | float32 | noncausal | {method} |" in text and f"| 32 | float32 | causal | {method} |" in text
    assert "_scaled_dot_product" in text  # the SDPA kernel the profiler observed
    assert "python -m attention_lab.benchmark --preset smoke" in text
    for kind in ("latency", "memory"):
        assert (tmp_path / "rep" / f"smoke-{kind}.png").stat().st_size > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_report_from_real_mps_run_has_no_memory_numbers(tmp_path):
    run = tmp_path / "mps.json"
    benchmark.main(["--device", "mps", "--seq-lens", "64", "--event-reps", "3", "--warmup", "1", "--output", str(run)])
    report.main([str(run), "--out-dir", str(tmp_path / "rep")])
    text = (tmp_path / "rep" / "report.md").read_text()
    assert "| n/a |" in text  # MPS memory is reported as unavailable, never estimated
    assert (tmp_path / "rep" / "mps-latency.png").exists()
    assert not (tmp_path / "rep" / "mps-memory.png").exists()
