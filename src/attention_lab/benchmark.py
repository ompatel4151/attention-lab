"""Inference-only benchmark of attention implementations on synthetic tensors.

    python -m attention_lab.benchmark --device cpu --preset smoke --output results/run.json

Methods: the explicit reference, PyTorch SDPA (automatic backend), our tiled
online-softmax attention (eager and torch.compile'd), and SDPA forced to one
backend with sdpa_kernel. This times one attention call; it says nothing about
whole-model (LLM) performance.

Timing boundary: only the call `method(q, k, v, causal=...)`. Input generation,
host->device copies, correctness checks, profiling, memory probes and file I/O
all happen outside it.
"""

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
import timeit
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

from .attention import reference_attention, sdpa_attention, tiled_attention, tiled_attention_compiled


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
    "tiled_compiled": tiled_attention_compiled,
    "sdpa_math": _forced(SDPBackend.MATH),
    "sdpa_flash": _forced(SDPBackend.FLASH_ATTENTION),
    "sdpa_efficient": _forced(SDPBackend.EFFICIENT_ATTENTION),
}
DEFAULT_METHODS = ["reference", "sdpa", "tiled"]
TILED = {"tiled", "tiled_compiled"}  # methods that take --block-size

PRESETS = {
    "smoke": dict(batch=1, heads=2, seq_lens=[32], head_dim=32, dtypes=["float32"]),
    "cpu-sweep": dict(batch=1, heads=2, seq_lens=[32, 64, 128, 256], head_dim=32, dtypes=["float32"]),
    # Large L leaves the overhead-dominated regime; calls take up to ~0.1 s, so measure longer.
    "cpu-large": dict(batch=1, heads=2, seq_lens=[512, 1024, 2048, 4096], head_dim=32, dtypes=["float32"],
                      min_run_time=1.5),
    "mps-sweep": dict(batch=1, heads=4, seq_lens=[256, 512, 1024, 2048, 4096], head_dim=64,
                      dtypes=["float32", "float16", "bfloat16"], device="mps"),
    "cuda-sweep": dict(batch=1, heads=4, seq_lens=[128, 256, 512, 1024], head_dim=64,
                       dtypes=["float32", "float16", "bfloat16"], device="cuda"),
}  # fmt: skip

# Allowed error of each method vs. "golden" = reference_attention in float64 on the CPU, from
# the same (already cast) inputs. float32: measured <= ~1e-6. float16 (eps 9.8e-4) and
# bfloat16 (eps 7.8e-3): bounds chosen from MPS measurements (see docs); NOT validated on CUDA.
TOLERANCES = {
    "float64": dict(atol=1e-10, rtol=1e-10),
    "float32": dict(atol=1e-5, rtol=1e-5),
    "float16": dict(atol=1e-2, rtol=1e-2),
    "bfloat16": dict(atol=5e-2, rtol=5e-2),
}

CSV_FIELDS = [
    "device", "dtype", "batch", "heads", "seq_len", "head_dim", "causal", "method", "status", "skip_reason",
    "median_s", "median_ci95_lo_s", "median_ci95_hi_s", "iqr_s", "min_s", "max_s", "num_samples", "rounds",
    "warmup_calls", "first_call_s", "max_abs_err", "correct", "speedup_vs_reference", "speedup_ci95_lo",
    "speedup_ci95_hi", "incremental_peak_bytes", "observed_sdpa_ops",
]  # fmt: skip

BOOTSTRAP_RESAMPLES = 1000


def speedup(reference_median: float, candidate_median: float) -> float:
    """> 1 means the candidate is faster than the reference."""
    if reference_median <= 0 or candidate_median <= 0:
        raise ValueError(f"medians must be positive, got {reference_median}, {candidate_median}")
    return reference_median / candidate_median


def _percentile_ci(values: list[float]) -> list[float]:
    values = sorted(values)
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def bootstrap_median_ci(samples: list[float], seed: int = 0) -> list[float]:
    """95% percentile-bootstrap interval for the median: resample with replacement, take the
    median each time, keep the 2.5th and 97.5th percentiles. Assumes samples are independent;
    timing samples drift and correlate, so treat it as a lower bound on the real uncertainty."""
    rng = random.Random(seed)
    return _percentile_ci([statistics.median(rng.choices(samples, k=len(samples))) for _ in range(BOOTSTRAP_RESAMPLES)])


def bootstrap_speedup_ci(reference: list[float], candidate: list[float], seed: int = 0) -> list[float]:
    """95% bootstrap interval for median(reference) / median(candidate), resampling both."""
    rng = random.Random(seed)
    ratios = [
        statistics.median(rng.choices(reference, k=len(reference)))
        / statistics.median(rng.choices(candidate, k=len(candidate)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _percentile_ci(ratios)


def summarize(samples: list[float]) -> dict:
    """Median (with bootstrap 95% CI) and spread of per-call times in seconds. IQR = p75 - p25."""
    q1, _, q3 = statistics.quantiles(samples, n=4) if len(samples) > 1 else samples * 3
    return {
        "median_s": statistics.median(samples),
        "median_ci95_s": bootstrap_median_ci(samples),
        "iqr_s": q3 - q1,
        "min_s": min(samples),
        "max_s": max(samples),
        "num_samples": len(samples),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_cpu(fn, warmup: int, min_run_time: float, threads: int) -> list[float]:
    """Per-call seconds, one sample per block of calls."""
    for _ in range(warmup):
        fn()
    # blocked_autorange picks a block size (calls per sample) so each block is long
    # enough to time reliably, then repeats blocks until min_run_time is used up.
    # num_threads: Timer defaults to 1 thread; we pass the real setting explicitly.
    # timer: Timer's default clock calls torch.accelerator.synchronize() on every read
    # when any accelerator exists (e.g. MPS on a Mac). CPU-only work needs no sync.
    m = Timer(stmt="fn()", globals={"fn": fn}, timer=timeit.default_timer, num_threads=threads)
    return m.blocked_autorange(min_run_time=min_run_time).times  # raw block time / calls per block


def time_events(fn, device: torch.device, warmup: int, reps: int) -> list[float]:
    """One call per sample, timed on the GPU with events (torch.cuda or torch.mps).

    GPU launches are asynchronous: fn() returns once work is queued, not done. Events are
    markers in the GPU's command stream; elapsed_time is GPU time between them.
    Run on MPS (Apple M4); the CUDA path shares this code but was never run on CUDA.
    """
    backend = torch.cuda if device.type == "cuda" else torch.mps
    for _ in range(warmup):
        fn()
    backend.synchronize()  # drain warmup work so it can't leak into the first sample
    samples = []
    for _ in range(reps):
        start, end = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        # Block the CPU until the GPU has finished everything queued, including `end`.
        # Device-wide, not end.synchronize(): on torch 2.14.1 MPS, end.synchronize() followed
        # by elapsed_time() hangs forever (reproduced in isolation); this order works.
        backend.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)  # elapsed_time is in ms
    return samples


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
    bytes alive before the call. Only PyTorch's CPU-allocator memory is seen.

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


def probe_method(fn, golden, tol, device) -> dict:
    """Everything except timing: first-call time, correctness, kernel names, memory."""
    t0 = time.perf_counter()
    out = fn()
    _synchronize(device)
    first_call = time.perf_counter() - t0  # includes torch.compile's compilation for compiled methods
    out = out.cpu().double()
    max_err = (out - golden).abs().max().item()
    correct = torch.allclose(out, golden, **tol)
    del out

    ops, cpu_peak = profile_call(fn)
    if device.type == "cuda":
        memory = peak_memory_cuda(fn)
    elif device.type == "cpu":
        memory = {"incremental_peak_bytes": cpu_peak}
    else:
        memory = None  # MPS has no peak-memory counter (only current/driver allocated); not estimated
    return {
        "status": "ok",
        "skip_reason": None,
        "first_call_s": first_call,
        "max_abs_err": max_err,
        "correct": correct,
        # Which SDPA kernel actually ran, as seen by the profiler (empty for our own methods).
        "observed_sdpa_ops": [o for o in ops if o.startswith("aten::_scaled_dot_product")],
        "memory": memory,
    }


def _reason(e: Exception) -> str:
    return f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"


def _skip(device, e) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return {"status": "skipped", "skip_reason": _reason(e)}


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
        cpu_inputs = [torch.randn(batch, heads, seq_len, head_dim, generator=g).to(dtype) for _ in range(3)]
        q, k, v = (x.to(device) for x in cpu_inputs)
        # inference_mode: no autograd graph is recorded, and outputs can never be used in backward.
        with torch.inference_mode():
            # Golden on the CPU: MPS has no float64. Built from the already-cast inputs, so it
            # measures each method's computation error, not the input rounding.
            golden = reference_attention(*(x.double() for x in cpu_inputs), causal=causal)
    except RuntimeError as e:  # includes torch.OutOfMemoryError
        return {**rec, **_skip(device, e)}

    fns = {
        name: partial(METHODS[name], q, k, v, causal=causal, **({"block_size": opts.block_size} if name in TILED else {}))
        for name in opts.methods
    }
    methods, samples = {}, {}
    with torch.inference_mode():
        for name, fn in fns.items():
            try:
                methods[name] = probe_method(fn, golden, tol, device)
                samples[name] = []
            except RuntimeError as e:  # a forced backend that can't run this input, OOM, unsupported dtype
                methods[name] = _skip(device, e)
        del golden

        # Timing, interleaved: each round times every method once, and the starting method
        # rotates, so slow drift (thermal, background load) and "who ran first" hit all methods.
        for rnd in range(opts.rounds):
            live = [n for n in samples if methods[n]["status"] == "ok"]
            for name in live[rnd % max(len(live), 1) :] + live[: rnd % max(len(live), 1)]:
                warmup = opts.warmup if rnd == 0 else 0
                try:
                    if device.type == "cpu":
                        samples[name] += time_cpu(fns[name], warmup, opts.min_run_time / opts.rounds, opts.threads)
                    else:
                        samples[name] += time_events(fns[name], device, warmup, max(opts.event_reps // opts.rounds, 1))
                except RuntimeError as e:
                    methods[name] = _skip(device, e)

    for name, s in samples.items():
        if methods[name]["status"] == "ok":
            methods[name]["timing"] = {
                **summarize(s), "rounds": opts.rounds, "warmup_calls": opts.warmup, "samples_s": s,
            }  # fmt: skip

    ok = {n for n, m in methods.items() if m["status"] == "ok"}
    speedups, cis = {}, {}
    if "reference" in ok:
        for n in sorted(ok - {"reference"}, key=opts.methods.index):
            speedups[n] = speedup(statistics.median(samples["reference"]), statistics.median(samples[n]))
            cis[n] = bootstrap_speedup_ci(samples["reference"], samples[n])
    return {
        **rec,
        "golden": "reference_attention in float64 on the CPU, same inputs",
        "tolerance": tol,
        "methods": methods,
        "speedup_vs_reference": speedups,
        "speedup_ci95": cis,
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"--device auto -> using {dev}")
        return dev
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"error: --device cuda requested, but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, CUDA build: {torch.version.cuda}). Not falling back."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("error: --device mps requested, but torch.backends.mps.is_available() is False. Not falling back.")
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
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "Apple MPS" if device.type == "mps" else None,
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
                    t = m["timing"]
                    row.update({k: t[k] for k in ("median_s", "iqr_s", "min_s", "max_s", "num_samples", "rounds", "warmup_calls")})
                    row["median_ci95_lo_s"], row["median_ci95_hi_s"] = t["median_ci95_s"]
                    row.update(first_call_s=m["first_call_s"], max_abs_err=m["max_abs_err"], correct=m["correct"])
                    row["speedup_vs_reference"] = r["speedup_vs_reference"].get(name)
                    row["speedup_ci95_lo"], row["speedup_ci95_hi"] = r["speedup_ci95"].get(name, (None, None))
                    row["incremental_peak_bytes"] = m["memory"]["incremental_peak_bytes"] if m["memory"] else None
                    row["observed_sdpa_ops"] = " ".join(m["observed_sdpa_ops"])
                w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="python -m attention_lab.benchmark", description=__doc__.split("\n\n")[0])
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], help="default: the preset's device, else cpu")
    p.add_argument("--preset", choices=PRESETS, default="smoke")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS, help="must include reference")
    p.add_argument("--block-size", type=int, default=64, help="key block size for the tiled methods")
    p.add_argument("--batch", type=int, help="override preset")
    p.add_argument("--heads", type=int, help="override preset")
    p.add_argument("--head-dim", type=int, help="override preset")
    p.add_argument("--seq-lens", type=int, nargs="+", help="override preset")
    p.add_argument("--dtypes", nargs="+", choices=TOLERANCES, help="override preset")
    p.add_argument("--causal", choices=["both", "on", "off"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10, help="untimed calls per method before the first round")
    p.add_argument("--rounds", type=int, default=3, help="interleaved timing rounds per configuration")
    p.add_argument("--min-run-time", type=float, help="CPU: total seconds of measurement per method (default: preset or 0.3)")
    p.add_argument("--event-reps", type=int, default=60, help="CUDA/MPS: total measured calls per method")
    p.add_argument("--threads", type=int, help="CPU threads (default: torch.get_num_threads())")
    p.add_argument("--output", type=Path, default=Path("results/run.json"))
    opts = p.parse_args(argv)

    cfg = dict(PRESETS[opts.preset])
    preset_device = cfg.pop("device", None)
    device = resolve_device(opts.device or preset_device or "cpu")
    if preset_device and device.type != preset_device:
        p.error(f"--preset {opts.preset} requires --device {preset_device}")
    if "reference" not in opts.methods:
        p.error("--methods must include reference (the baseline for speedups)")
    if opts.block_size < 1 or opts.rounds < 1:
        p.error("--block-size and --rounds must be >= 1")
    if opts.threads:
        torch.set_num_threads(opts.threads)
    opts.threads = torch.get_num_threads()

    for key in ("batch", "heads", "head_dim", "seq_lens", "dtypes"):
        if getattr(opts, key) is not None:
            cfg[key] = getattr(opts, key)
    preset_min_run_time = cfg.pop("min_run_time", 0.3)
    opts.min_run_time = opts.min_run_time or preset_min_run_time
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
        "rounds": opts.rounds,
        "timing_method": (
            f"torch.{device.type}.Event pairs around 1 call, {opts.event_reps} samples over {opts.rounds} interleaved rounds"
            if device.type != "cpu"
            else f"torch.utils.benchmark.Timer.blocked_autorange(min_run_time={opts.min_run_time / opts.rounds:g} per round), "
            f"{opts.rounds} interleaved rounds, timer=timeit.default_timer, num_threads={opts.threads}; "
            "sample = mean per-call time of one block"
        ),
        "time_unit": "seconds",
        "confidence_intervals": f"95% percentile bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed 0",
        "memory_method": {
            "cuda": "torch.cuda.reset_peak_memory_stats + max_memory_allocated per method (allocated, not reserved)",
            "cpu": "torch.profiler(profile_memory=True): peak of live CPU-allocator bytes during one call, "
            "above the bytes alive before it",
            "mps": "unavailable: MPS exposes no peak-memory counter",
        }[device.type],
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
                        print(f"{tag} {name:<15} SKIPPED: {m['skip_reason']}")
                        continue
                    t, ratio = m["timing"], r["speedup_vs_reference"].get(name)
                    peak = f"{m['memory']['incremental_peak_bytes'] / 2**20:8.2f} MiB" if m["memory"] else "     n/a    "
                    print(
                        f"{tag} {name:<15} {t['median_s'] * 1e6:10.1f} us (IQR {t['iqr_s'] * 1e6:7.1f})  peak {peak}"
                        f"  {'ref/this ' + format(ratio, '.2f') + 'x' if ratio else '':<15} err {m['max_abs_err']:.1e}"
                        f" correct={m['correct']}"
                    )

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps({"environment": env, "settings": settings, "results": results}, indent=2))
    csv_path = opts.output.with_suffix(".csv")
    write_csv(csv_path, env, results)
    print(f"wrote {opts.output} and {csv_path}")

    failed = [
        (r["seq_len"], r["dtype"], r["causal"], n)
        for r in results if r["status"] == "ok"
        for n, m in r["methods"].items() if m["status"] == "ok" and not m["correct"]
    ]  # fmt: skip
    if failed:
        print(f"WARNING: correctness failed for {failed}; timings recorded but suspect.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
