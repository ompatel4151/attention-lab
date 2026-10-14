"""Build a Markdown report and plots from saved benchmark JSON files.

    python -m attention_lab.report                                  # every results/*.json
    python -m attention_lab.report results/cpu-sweep.json --out-dir reports

Reads real saved results only. Each JSON file is one environment and gets its own
section and plots; runs from different machines/devices are never merged on one axis.
With no input files the report says so instead of showing any numbers.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file-only backend: works headless and never opens a window
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
from matplotlib.ticker import FuncFormatter  # noqa: E402

COLORS = {
    "reference": "tab:blue", "sdpa": "tab:orange", "tiled": "tab:green",
    "sdpa_math": "tab:red", "sdpa_flash": "tab:purple", "sdpa_efficient": "tab:brown",
}  # fmt: skip
MIB = 2**20


def us(seconds: float) -> float:
    return seconds * 1e6


def causal_name(causal: bool) -> str:
    return "causal" if causal else "noncausal"


def environment_label(env: dict) -> str:
    hw = env["gpu_name"] or env["machine"]
    return f"{env['device']} ({hw}), torch {env['torch']}, {env['torch_num_threads']} CPU threads"


def ok_methods(r: dict) -> dict:
    return {n: m for n, m in r["methods"].items() if m["status"] == "ok"} if r["status"] == "ok" else {}


def series(run: dict) -> dict:
    """{(method, dtype, causal): [(seq_len, method_result), ...] sorted by seq_len} for successful methods."""
    out = defaultdict(list)
    for r in run["results"]:
        for method, m in ok_methods(r).items():
            out[(method, r["dtype"], r["causal"])].append((r["seq_len"], m))
    return {k: sorted(v, key=lambda p: p[0]) for k, v in out.items()}


def nominal_flops(r: dict) -> int:
    """Multiply-adds of the two matmuls (QK^T and PV), x2 for FLOPs; causal counted as full."""
    return 4 * r["batch"] * r["heads"] * r["seq_len"] ** 2 * r["head_dim"]


def _style_axes(ax, seq_lens, ylabel, title, log_y):
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens, [str(L) for L in seq_lens])
    ax.minorticks_off()
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("sequence length L (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="major", alpha=0.3)


def _plot(run, path, values, ylabel, title, force_log=False):
    """values(method_result) -> (y, (low, high) or None). One panel per dtype; colour = method,
    dashed = causal; one legend outside the panels. Log y when the data spans > 30x."""
    data = series(run)
    dtypes = list(dict.fromkeys(dtype for _, dtype, _ in data))
    fig, axes = plt.subplots(len(dtypes), 1, figsize=(8.5, 3.6 * len(dtypes) + 1.0), squeeze=False)
    for ax, dtype in zip(axes[:, 0], dtypes):
        ys = []
        for (method, dt, causal), pts in data.items():
            if dt != dtype:
                continue
            vals = [values(m) for _, m in pts]
            y = [v[0] for v in vals]
            ys += y
            yerr = None
            if all(v[1] for v in vals):
                yerr = [[yy - v[1][0] for yy, v in zip(y, vals)], [v[1][1] - yy for yy, v in zip(y, vals)]]
            ax.errorbar(
                [L for L, _ in pts], y, yerr=yerr, capsize=3, color=COLORS.get(method, "black"), marker="o",
                markersize=4, linestyle="--" if causal else "-", label=f"{method}, {causal_name(causal)}",
            )  # fmt: skip
        seq_lens = sorted({L for (_, dt, _), pts in data.items() if dt == dtype for L, _ in pts})
        log_y = force_log or max(ys) / max(min(ys), 1e-12) > 30
        _style_axes(ax, seq_lens, ylabel, dtype, log_y)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_latency(run: dict, path: Path) -> None:
    def values(m):
        lo, hi = m["timing"]["median_ci95_s"]
        return us(m["timing"]["median_s"]), (us(lo), us(hi))  # error bars: bootstrap 95% CI of the median

    title = f"Attention latency (median, bars = 95% CI of the median)\n{environment_label(run['environment'])}"
    _plot(run, path, values, "latency per call (µs)", title)


def has_memory(run: dict) -> bool:
    return any(m["memory"] for pts in series(run).values() for _, m in pts)


def plot_memory(run: dict, path: Path) -> None:
    title = (
        f"Incremental peak allocated memory per call (log-log: slope 2 = L², slope 1 = L)\n"
        f"{environment_label(run['environment'])}"
    )
    _plot(run, path, lambda m: (m["memory"]["incremental_peak_bytes"] / MIB, None), "peak above baseline (MiB)",
          title, force_log=True)  # fmt: skip


def results_table(run: dict) -> list[str]:
    head = (
        "| L | dtype | mask | method | median µs [95% CI] | ref / method [95% CI] | peak MiB | first call ms "
        "| max abs err | correct | SDPA kernel observed |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    blanks = " |" * 7
    for r in run["results"]:
        prefix = f"| {r['seq_len']} | {r['dtype']} | {causal_name(r['causal'])} |"
        if r["status"] != "ok":
            lines.append(f"{prefix} all | skipped: {r['skip_reason']} |{blanks}")
            continue
        for name, m in r["methods"].items():
            if m["status"] != "ok":
                lines.append(f"{prefix} {name} | skipped: {m['skip_reason']} |{blanks}")
                continue
            t, ratio = m["timing"], r["speedup_vs_reference"].get(name)
            ci = r["speedup_ci95"].get(name)
            lo, hi = t["median_ci95_s"]
            ratio_txt = f"{ratio:.2f}x [{ci[0]:.2f}, {ci[1]:.2f}]" if ratio else "-"
            peak_txt = f"{m['memory']['incremental_peak_bytes'] / MIB:.2f}" if m["memory"] else "n/a"
            lines.append(
                f"{prefix} {name} | {us(t['median_s']):.1f} [{us(lo):.1f}, {us(hi):.1f}] | {ratio_txt} | {peak_txt} | "
                f"{m['first_call_s'] * 1e3:.1f} | {m['max_abs_err']:.1e} | {'yes' if m['correct'] else '**NO**'} | "
                f"{', '.join(o.removeprefix('aten::') for o in m['observed_sdpa_ops']) or '-'} |"
            )
    return lines


def observations(run: dict) -> list[str]:
    """Statements computed from this run's numbers only."""
    ok = [r for r in run["results"] if ok_methods(r)]
    if not ok:
        return ["- No successful configurations in this run; nothing to interpret."]
    lines = []
    ratios = defaultdict(list)
    for r in ok:
        for name, ratio in r["speedup_vs_reference"].items():
            ratios[(name, r["dtype"], r["causal"])].append((r["seq_len"], ratio, r["speedup_ci95"][name]))
    for (name, dtype, causal), pts in ratios.items():
        # "Faster"/"slower" only when the whole 95% CI of the ratio is on one side of 1.
        faster = [L for L, _, ci in pts if ci[0] > 1]
        slower = [L for L, _, ci in pts if ci[1] < 1]
        lines.append(
            f"- {name} vs reference, {dtype}, {causal_name(causal)}: reference/{name} ratio "
            f"{min(x for _, x, _ in pts):.2f}x to {max(x for _, x, _ in pts):.2f}x over L = "
            f"{', '.join(str(L) for L, _, _ in pts)}. 95% CI above 1 (faster) at L = "
            f"{', '.join(map(str, faster)) or 'none'}; below 1 (slower) at L = {', '.join(map(str, slower)) or 'none'}."
        )
    kernels = defaultdict(set)
    for r in ok:
        for name, m in ok_methods(r).items():
            kernels[name].update(m["observed_sdpa_ops"])
    for name, ops in kernels.items():
        if ops:
            lines.append(f"- {name}: profiler saw {', '.join(sorted(ops))}.")
    failed = [(r["seq_len"], n) for r in ok for n, m in ok_methods(r).items() if not m["correct"]]
    if failed:
        lines.append(f"- **Correctness failed for {failed}; those timings are suspect.**")
    skipped = [f"{n} at L={r['seq_len']} {causal_name(r['causal'])}" for r in run["results"] if r["status"] == "ok"
               for n, m in r["methods"].items() if m["status"] != "ok"]  # fmt: skip
    skipped += [f"all methods at L={r['seq_len']} {causal_name(r['causal'])}" for r in run["results"] if r["status"] != "ok"]
    if skipped:
        lines.append(f"- Skipped (reasons in the table): {'; '.join(skipped)}.")

    growth = []
    for (method, dtype, causal), pts in series(run).items():
        if len(pts) >= 2:
            (L0, m0), (L1, m1) = pts[0], pts[-1]
            t = m1["timing"]["median_s"] / m0["timing"]["median_s"]
            mem = (
                f"x{m1['memory']['incremental_peak_bytes'] / m0['memory']['incremental_peak_bytes']:.1f}"
                if m0["memory"] and m1["memory"]
                else "n/a"
            )
            growth.append(
                f"| {method} | {dtype} | {causal_name(causal)} | {L0} → {L1} | x{t:.1f} | {mem} "
                f"| x{L1 / L0:g} | x{(L1 / L0) ** 2:g} |"
            )
    if growth:
        lines += [
            "",
            "Growth across the measured range, compared with linear and quadratic growth in L:",
            "",
            "| method | dtype | mask | L range | latency growth | peak-memory growth | L growth | L² growth |",
            "|---|---|---|---|---|---|---|---|",
            *growth,
        ]

    L_max = max(r["seq_len"] for r in ok)
    tput = [(r, n, m) for r in ok if r["seq_len"] == L_max and not r["causal"] for n, m in ok_methods(r).items()]
    if tput:
        lines += [
            "",
            f"Nominal throughput at L = {L_max}, noncausal (4·B·H·L²·D FLOPs for the two matmuls ÷ median latency; "
            "softmax and masking not counted):",
            "",
            "| method | dtype | GFLOP/s |",
            "|---|---|---|",
            *(f"| {n} | {r['dtype']} | {nominal_flops(r) / m['timing']['median_s'] / 1e9:.1f} |" for r, n, m in tput),
        ]
    return lines


CAVEATS = """\
How to read this:
- One run on one machine. Re-run before quoting numbers; compare only rows from the same environment.
- Timings cover one attention call including the identical Python validation in every wrapper;
  input creation, correctness checks, profiling and I/O are excluded. This is not whole-model performance.
- SDPA's backend is chosen automatically; the kernel the profiler saw is listed per row. Forced-backend
  methods (sdpa_math, sdpa_flash, ...) also pay for entering the sdpa_kernel context on every call.
- The score matrix is O(L²), but at small L fixed per-call costs (Python, dispatch, allocation) dominate,
  so latency can grow much more slowly than L².
- The explicit causal reference builds an [L, L] mask on every call. The tiled method loops over key blocks
  in Python, so its per-block overhead is large at small L, and causal does not skip any blocks.
- CPU peak memory comes from torch.profiler allocation events and counts only PyTorch-allocator memory.
  MPS has no peak-memory counter, so MPS memory is n/a (not estimated).
- Confidence intervals are percentile bootstraps over timing samples, which assumes independent samples;
  drift and correlation make the real uncertainty larger. "First call" includes torch.compile's compile time."""


def build_report(json_paths: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Attention Lab benchmark report", ""]
    if not json_paths:
        md += [
            "**No benchmark results found.** Nothing is reported; no sample numbers are shown.",
            "",
            "Create results first, e.g.:",
            "",
            "```bash",
            "python -m attention_lab.benchmark --device cpu --preset cpu-sweep --output results/cpu-sweep.json",
            "```",
        ]
    for path in json_paths:
        run = json.loads(path.read_text())
        env, st = run["environment"], run["settings"]
        md += [
            f"## `{path.name}`",
            "",
            f"- Environment: {environment_label(env)}; Python {env['python']}; {env['os']}; "
            f"CUDA runtime {env['cuda_runtime']}; recorded {env['timestamp_utc']}",
            f"- Shapes: B={st['batch']}, H={st['heads']}, D={st['head_dim']}, L in {st['seq_lens']}; "
            f"dtypes {st['dtypes']}; seed {st['seed']}",
            f"- Methods: {', '.join(st['methods'])}; tiled block size {st['tiled_block_size']}",
            f"- Timing: {st['timing_method']}; {st['warmup_calls']} warmup calls per method; unit {st['time_unit']}",
            f"- Memory: {st['memory_method']}",
            f"- SDPA backend: {st['sdpa_backend']}",
            f"- Correctness: each method vs float64 reference; tolerances {st['tolerances']}",
            "",
            "Reproduce (from the project root, inside the venv):",
            "",
            "```bash",
            "python -m attention_lab.benchmark " + " ".join(st["argv"]),
            "```",
            "",
            *results_table(run),
            "",
        ]
        if series(run):
            for kind, plot in (("latency", plot_latency), ("memory", plot_memory)):
                if kind == "memory" and not has_memory(run):
                    continue
                png = out_dir / f"{path.stem}-{kind}.png"
                plot(run, png)
                md += [f"![{kind} vs L]({png.name})", ""]
        md += ["### Observations (computed from this run)", "", *observations(run), "", CAVEATS, ""]

    report = out_dir / "report.md"
    report.write_text("\n".join(md))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m attention_lab.report", description=__doc__.split("\n\n")[0])
    p.add_argument("inputs", nargs="*", type=Path, help="benchmark JSON files (default: results/*.json)")
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    opts = p.parse_args(argv)
    inputs = opts.inputs or sorted(Path("results").glob("*.json"))
    report = build_report(inputs, opts.out_dir)
    print(f"wrote {report} from {len(inputs)} result file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
