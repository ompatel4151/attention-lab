"""Inference-only benchmark of attention implementations on synthetic tensors.

    python -m attention_lab.benchmark --device cpu --preset smoke --output results/run.json

Methods: the explicit reference, PyTorch SDPA (automatic backend), our tiled
online-softmax attention, and SDPA forced to one backend with sdpa_kernel.
This times one attention call; it says nothing about whole-model (LLM) performance.

Timing boundary: only the call `method(q, k, v, causal=...)`. Input generation,
host->device copies, correctness checks, profiling, memory probes and file I/O
all happen outside it.
"""

import argparse
import csv
import json
import platform
import statistics
import sys
import timeit
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

from .attention import reference_attention, sdpa_attention, tiled_attention


def _forced(backend: SDPBackend):
    """sdpa_attention restricted to one backend; raises RuntimeError if it can't run the input."""

    def fn(q, k, v, causal=False):
        with sdpa_kernel([backend]):
            return sdpa_attention(q, k, v, causal=causal)

    return fn


METHODS = {
    "reference": reference_attention,
    "sdpa": sdpa_attention,
    "tiled": tiled_attention,
    "sdpa_math": _forced(SDPBackend.MATH),
    "sdpa_flash": _forced(SDPBackend.FLASH_ATTENTION),
    "sdpa_efficient": _forced(SDPBackend.EFFICIENT_ATTENTION),
}
DEFAULT_METHODS = ["reference", "sdpa", "tiled"]

PRESETS = {
    "smoke": dict(batch=1, heads=2, seq_lens=[32], head_dim=32, dtypes=["float32"]),
    "cpu-sweep": dict(batch=1, heads=2, seq_lens=[32, 64, 128, 256], head_dim=32, dtypes=["float32"]),
    # Large L leaves the overhead-dominated regime; calls take up to ~0.1 s, so measure longer.
    "cpu-large": dict(batch=1, heads=2, seq_lens=[512, 1024, 2048, 4096], head_dim=32, dtypes=["float32"],
                      min_run_time=1.0),
    "cuda-sweep": dict(
        batch=1, heads=4, seq_lens=[128, 256, 512, 1024], head_dim=64, dtypes=["float32", "float16", "bfloat16"]
    ),
}  # fmt: skip

# Allowed error of each method vs. "golden" = reference_attention on the same inputs
# upcast to float64. float32: measured <= ~2e-6 on CPU. float16 (eps 9.8e-4) and
# bfloat16 (eps 7.8e-3) bounds are starting guesses, NOT yet validated on a GPU.
TOLERANCES = {
    "float64": dict(atol=1e-10, rtol=1e-10),
    "float32": dict(atol=1e-5, rtol=1e-5),
    "float16": dict(atol=1e-2, rtol=1e-2),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

CSV_FIELDS = [
    "device", "dtype", "batch", "heads", "seq_len", "head_dim", "causal", "method", "status", "skip_reason",
    "median_s", "iqr_s", "min_s", "max_s", "num_samples", "calls_per_sample", "warmup_calls",
    "max_abs_err", "correct", "speedup_vs_reference", "incremental_peak_bytes", "observed_sdpa_ops",
]  # fmt: skip


def speedup(reference_median: float, candidate_median: float) -> float:
    """> 1 means the candidate is faster than the reference."""
    if reference_median <= 0 or candidate_median <= 0:
        raise ValueError(f"medians must be positive, got {reference_median}, {candidate_median}")
    return reference_median / candidate_median


def summarize(samples: list[float]) -> dict:
    """Median and spread of per-call times in seconds. IQR = p75 - p25."""
    q1, _, q3 = statistics.quantiles(samples, n=4) if len(samples) > 1 else samples * 3
    return {
        "median_s": statistics.median(samples),
        "iqr_s": q3 - q1,
        "min_s": min(samples),
        "max_s": max(samples),
        "num_samples": len(samples),
    }


def time_cpu(fn, warmup: int, min_run_time: float, threads: int) -> tuple[list[float], int]:
    """Returns (per-call seconds for each block, calls per block)."""
    for _ in range(warmup):
        fn()
    # blocked_autorange picks a block size (calls per sample) so each block is long
    # enough to time reliably, then repeats blocks until min_run_time is used up.
    # num_threads: Timer defaults to 1 thread; we pass the real setting explicitly.
    # timer: Timer's default clock calls torch.accelerator.synchronize() on every read
    # when any accelerator exists (e.g. MPS on a Mac). CPU-only work needs no sync.
    m = Timer(stmt="fn()", globals={"fn": fn}, timer=timeit.default_timer, num_threads=threads)
    m = m.blocked_autorange(min_run_time=min_run_time)
    return m.times, m.number_per_run  # m.times = raw block time / calls per block


def time_cuda(fn, warmup: int, reps: int) -> tuple[list[float], int]:
    """One call per sample, timed on the GPU with CUDA events. NOT run on the author's machine."""
    for _ in range(warmup):
        fn()
    # CUDA launches are asynchronous: fn() returns once work is queued, not done.
    # Drain the warmup work so it can't leak into the first measured sample.
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()  # marker enqueued on the current stream
        fn()
        end.record()
        end.synchronize()  # block the CPU until the GPU has passed `end`
        samples.append(start.elapsed_time(end) / 1e3)  # elapsed_time is in ms
    return samples, 1


def peak_memory_cuda(fn) -> dict:
    """Allocated (not reserved) bytes for one call. NOT run on the author's machine.

    baseline = tensors already alive (q, k, v, ...). incremental = extra memory the
    call needed at its peak. The caching allocator usually *reserves* more than is
    allocated; that is torch.cuda.memory_reserved, not reported.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return {"baseline_allocated_bytes": baseline, "peak_allocated_bytes": peak, "incremental_peak_bytes": peak - baseline}


def profile_call(fn) -> tuple[list[str], int]:
    """Run fn once under torch.profiler. Returns (aten ops in call order, CPU incremental peak bytes).

    Peak: every allocation (+bytes) and free (-bytes) is a raw "[memory]" record with its own
    timestamp. Summing them in time order gives live bytes; the max is the peak above the
    bytes alive before the call. Only PyTorch-allocator memory is seen.

    Per-op totals (FunctionEvent.self_cpu_memory_usage) are NOT usable for this: a free
    inside an op (e.g. SDPA's math kernel releasing its [L, L] buffer) is charged to the
    op as a whole, losing *when* it happened, and hid that kernel's peak in an earlier
    version. `prof.profiler.kineto_results` is not a documented API (checked on torch 2.14.1);
    test_benchmark.py guards it.
    """
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        out = fn()
        del out
    records = sorted((e.start_ns(), e.nbytes()) for e in prof.profiler.kineto_results.events() if e.name() == "[memory]")
    live = peak = 0
    for _, nbytes in records:
        live += nbytes
        peak = max(peak, live)
    events = sorted(prof.events(), key=lambda e: e.time_range.start)
    return list(dict.fromkeys(e.name for e in events if e.name.startswith("aten::"))), peak


def measure_method(fn, golden, tol, device, opts) -> dict:
    out = fn().double()
    max_err = (out - golden).abs().max().item()
    correct = torch.allclose(out, golden, **tol)
    del out

    ops, cpu_peak = profile_call(fn)
    if device.type == "cuda":
        samples, per = time_cuda(fn, opts.warmup, opts.cuda_reps)
        memory = peak_memory_cuda(fn)
    else:
        samples, per = time_cpu(fn, opts.warmup, opts.min_run_time, opts.threads)
        memory = {"incremental_peak_bytes": cpu_peak}
    return {
        "status": "ok",
        "skip_reason": None,
        "max_abs_err": max_err,
        "correct": correct,
        # Which SDPA kernel actually ran, as seen by the profiler (empty for our own methods).
        "observed_sdpa_ops": [o for o in ops if o.startswith("aten::_scaled_dot_product")],
        "timing": {**summarize(samples), "calls_per_sample": per, "warmup_calls": opts.warmup, "samples_s": samples},
        "memory": memory,
    }


def _reason(e: Exception) -> str:
    return f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"


def run_config(device, dtype_name, batch, heads, seq_len, head_dim, causal, opts) -> dict:
    rec = dict(
        dtype=dtype_name, batch=batch, heads=heads, seq_len=seq_len, head_dim=head_dim, causal=causal,
        status="ok", skip_reason=None,
    )  # fmt: skip
    dtype = getattr(torch, dtype_name)
    if device.type == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        return {**rec, "status": "skipped", "skip_reason": "bfloat16 not natively supported on this GPU"}

    tol = TOLERANCES[dtype_name]
    try:
        # Same seed -> same values on every device: generate on CPU in float32, then move/cast.
        g = torch.Generator().manual_seed(opts.seed)
        q, k, v = (torch.randn(batch, heads, seq_len, head_dim, generator=g).to(device, dtype) for _ in range(3))
        # inference_mode: no autograd graph is recorded, and outputs can never be used in backward.
        with torch.inference_mode():
            golden = reference_attention(q.double(), k.double(), v.double(), causal=causal)
    except RuntimeError as e:  # includes torch.OutOfMemoryError
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {**rec, "status": "skipped", "skip_reason": _reason(e)}

    methods = {}
    with torch.inference_mode():
        for name in opts.methods:
            kwargs = {"block_size": opts.block_size} if name == "tiled" else {}
            fn = partial(METHODS[name], q, k, v, causal=causal, **kwargs)
            try:
                methods[name] = measure_method(fn, golden, tol, device, opts)
            except RuntimeError as e:  # a forced backend that can't run this input, OOM, unsupported dtype
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                methods[name] = {"status": "skipped", "skip_reason": _reason(e)}
    del golden

    ok = {n: m for n, m in methods.items() if m["status"] == "ok"}
    ref = ok.get("reference")
    speedups = (
        {n: speedup(ref["timing"]["median_s"], m["timing"]["median_s"]) for n, m in ok.items() if n != "reference"}
        if ref
        else {}
    )
    return {
        **rec,
        "golden": "reference_attention in float64 on the same inputs",
        "tolerance": tol,
        "methods": methods,
        "speedup_vs_reference": speedups,
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--device auto -> using {dev}")
        return dev
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"error: --device cuda requested, but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, CUDA build: {torch.version.cuda}). Not falling back."
        )
    return torch.device(name)


def environment(device: torch.device) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "os": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
    }


def write_csv(path: Path, env: dict, results: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            base = {k: r[k] for k in ("dtype", "batch", "heads", "seq_len", "head_dim", "causal", "status", "skip_reason")}
            base["device"] = env["device"]
            if r["status"] != "ok":
                w.writerow(base)
                continue
            for name, m in r["methods"].items():
                row = {**base, "method": name, "status": m["status"], "skip_reason": m["skip_reason"]}
                if m["status"] == "ok":
                    row.update({k: v for k, v in m["timing"].items() if k != "samples_s"})
                    row.update(max_abs_err=m["max_abs_err"], correct=m["correct"])
                    row["speedup_vs_reference"] = r["speedup_vs_reference"].get(name)
                    row["incremental_peak_bytes"] = m["memory"]["incremental_peak_bytes"]
                    row["observed_sdpa_ops"] = " ".join(m["observed_sdpa_ops"])
                w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="python -m attention_lab.benchmark", description=__doc__.split("\n\n")[0])
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--preset", choices=PRESETS, default="smoke")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS, help="must include reference")
    p.add_argument("--block-size", type=int, default=64, help="key block size for the tiled method")
    p.add_argument("--batch", type=int, help="override preset")
    p.add_argument("--heads", type=int, help="override preset")
    p.add_argument("--head-dim", type=int, help="override preset")
    p.add_argument("--seq-lens", type=int, nargs="+", help="override preset")
    p.add_argument("--dtypes", nargs="+", choices=TOLERANCES, help="override preset")
    p.add_argument("--causal", choices=["both", "on", "off"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10, help="untimed calls per method before measuring")
    p.add_argument("--min-run-time", type=float, help="CPU: seconds of measurement per method (default: preset or 0.2)")
    p.add_argument("--cuda-reps", type=int, default=50, help="CUDA: measured calls per method")
    p.add_argument("--threads", type=int, help="CPU threads (default: torch.get_num_threads())")
    p.add_argument("--output", type=Path, default=Path("results/run.json"))
    opts = p.parse_args(argv)

    device = resolve_device(opts.device)
    if opts.preset == "cuda-sweep" and device.type != "cuda":
        p.error("--preset cuda-sweep requires a CUDA device")
    if "reference" not in opts.methods:
        p.error("--methods must include reference (the baseline for speedups)")
    if opts.block_size < 1:
        p.error("--block-size must be >= 1")
    if opts.threads:
        torch.set_num_threads(opts.threads)
    opts.threads = torch.get_num_threads()

    cfg = dict(PRESETS[opts.preset])
    for key in ("batch", "heads", "head_dim", "seq_lens", "dtypes"):
        if getattr(opts, key) is not None:
            cfg[key] = getattr(opts, key)
    opts.min_run_time = opts.min_run_time or cfg.pop("min_run_time", 0.2)
    cfg.pop("min_run_time", None)
    causal_flags = {"both": [False, True], "on": [True], "off": [False]}[opts.causal]

    env = environment(device)
    settings = {
        "argv": argv,
        "preset": opts.preset,
        **cfg,
        "causal": causal_flags,
        "methods": opts.methods,
        "tiled_block_size": opts.block_size,
        "seed": opts.seed,
        "warmup_calls": opts.warmup,
        "timing_method": (
            f"torch.cuda.Event pairs around 1 call, {opts.cuda_reps} samples"
            if device.type == "cuda"
            else f"torch.utils.benchmark.Timer.blocked_autorange(min_run_time={opts.min_run_time}), "
            f"timer=timeit.default_timer, num_threads={opts.threads}; sample = mean per-call time of one block"
        ),
        "time_unit": "seconds",
        "memory_method": (
            "torch.cuda.reset_peak_memory_stats + max_memory_allocated per method (allocated, not reserved)"
            if device.type == "cuda"
            else "torch.profiler(profile_memory=True): peak of live CPU-allocator bytes during one call, "
            "above the bytes alive before it"
        ),
        "sdpa_backend": "automatic selection; the kernel that ran is recorded per method as observed_sdpa_ops",
        "tolerances": {d: TOLERANCES[d] for d in cfg["dtypes"]},
    }

    print(f"device={device} torch={torch.__version__} threads={opts.threads} preset={opts.preset} methods={opts.methods}")
    results = []
    for seq_len in cfg["seq_lens"]:
        for dtype_name in cfg["dtypes"]:
            for causal in causal_flags:
                r = run_config(device, dtype_name, cfg["batch"], cfg["heads"], seq_len, cfg["head_dim"], causal, opts)
                results.append(r)
                tag = f"L={seq_len:<5} {dtype_name:<8} causal={causal!s:<5}"
                if r["status"] != "ok":
                    print(f"{tag} SKIPPED: {r['skip_reason']}")
                    continue
                for name, m in r["methods"].items():
                    if m["status"] != "ok":
                        print(f"{tag} {name:<14} SKIPPED: {m['skip_reason']}")
                        continue
                    t = m["timing"]
                    ratio = r["speedup_vs_reference"].get(name)
                    print(
                        f"{tag} {name:<14} {t['median_s'] * 1e6:10.1f} us (IQR {t['iqr_s'] * 1e6:7.1f})"
                        f"  peak {m['memory']['incremental_peak_bytes'] / 2**20:8.2f} MiB"
                        f"  {'ref/this ' + format(ratio, '.2f') + 'x' if ratio else '':<15} correct={m['correct']}"
                    )

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps({"environment": env, "settings": settings, "results": results}, indent=2))
    csv_path = opts.output.with_suffix(".csv")
    write_csv(csv_path, env, results)
    print(f"wrote {opts.output} and {csv_path}")

    failed = [
        (r["seq_len"], r["causal"], n)
        for r in results if r["status"] == "ok"
        for n, m in r["methods"].items() if m["status"] == "ok" and not m["correct"]
    ]  # fmt: skip
    if failed:
        print(f"WARNING: correctness failed for {failed}; timings recorded but suspect.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
