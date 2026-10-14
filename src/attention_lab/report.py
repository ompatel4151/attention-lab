"""Build a Markdown report and plots from saved benchmark JSON files.

    python -m attention_lab.report                                  # every results/*.json
    python -m attention_lab.report results/cpu-sweep.json --out-dir reports

Reads real saved results only. Each JSON file is one environment and gets its own
section and plots; runs from different machines/devices are never merged on one axis.
With no input files the report says so instead of showing any numbers.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file-only backend: works headless and never opens a window
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

COLORS = {"reference": "tab:blue", "sdpa": "tab:orange"}
MARKERS = {"float64": "s", "float32": "o", "float16": "^", "bfloat16": "v"}


def us(seconds: float) -> float:
    return seconds * 1e6


def causal_name(causal: bool) -> str:
    return "causal" if causal else "noncausal"


def environment_label(env: dict) -> str:
    hw = env["gpu_name"] or env["machine"]
    return f"{env['device']} ({hw}), torch {env['torch']}, {env['torch_num_threads']} CPU threads"


def series(run: dict) -> dict:
    """{(method, dtype, causal): [(seq_len, method_result), ...] sorted by seq_len} for successful configs."""
    out = defaultdict(list)
    for r in run["results"]:
        if r["status"] == "ok":
            for method, m in r["methods"].items():
                out[(method, r["dtype"], r["causal"])].append((r["seq_len"], m))
    return {k: sorted(v, key=lambda p: p[0]) for k, v in out.items()}


def _style_axes(ax, seq_lens, ylabel, title):
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens, [str(L) for L in seq_lens])
    ax.minorticks_off()
    ax.set_xlabel("sequence length L (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_latency(run: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (method, dtype, causal), pts in series(run).items():
        seq_lens = [L for L, _ in pts]
        med = [us(m["timing"]["median_s"]) for _, m in pts]
        # Error bars: p25..p75 of the raw samples (asymmetric, so computed from samples, not IQR).
        quart = [statistics.quantiles(m["timing"]["samples_s"], n=4) if len(m["timing"]["samples_s"]) > 1
                 else [m["timing"]["median_s"]] * 3 for _, m in pts]  # fmt: skip
        yerr = [[md - us(q[0]) for md, q in zip(med, quart)], [us(q[2]) - md for md, q in zip(med, quart)]]
        ax.errorbar(
            seq_lens, med, yerr=yerr, capsize=3, color=COLORS[method], marker=MARKERS[dtype],
            linestyle="--" if causal else "-", label=f"{method}, {dtype}, {causal_name(causal)}",
        )  # fmt: skip
    ax.set_ylim(bottom=0)
    all_L = sorted({r["seq_len"] for r in run["results"] if r["status"] == "ok"})
    title = f"Attention latency (median, bars = p25-p75)\n{environment_label(run['environment'])}"
    _style_axes(ax, all_L, "latency per call (µs)", title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_memory(run: dict, path: Path) -> bool:
    """CUDA runs only. Returns False (no file) when the run has no memory data."""
    data = {k: [(L, m["memory"]["incremental_peak_bytes"] / 2**20) for L, m in pts if m["memory"]]
            for k, pts in series(run).items()}  # fmt: skip
    data = {k: v for k, v in data.items() if v}
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (method, dtype, causal), pts in data.items():
        ax.plot(
            [L for L, _ in pts], [mib for _, mib in pts], color=COLORS[method], marker=MARKERS[dtype],
            linestyle="--" if causal else "-", label=f"{method}, {dtype}, {causal_name(causal)}",
        )  # fmt: skip
    ax.set_ylim(bottom=0)
    all_L = sorted({L for pts in data.values() for L, _ in pts})
    title = f"Incremental peak allocated memory per call\n{environment_label(run['environment'])}"
    _style_axes(ax, all_L, "max_memory_allocated - baseline (MiB)", title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def results_table(run: dict) -> list[str]:
    has_mem = run["environment"]["device"].startswith("cuda")
    head = "| L | dtype | causal | reference µs (IQR) | SDPA µs (IQR) | ref/SDPA | max abs err ref / SDPA | correct |"
    if has_mem:
        head += " incr. peak MiB ref / SDPA |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in run["results"]:
        prefix = f"| {r['seq_len']} | {r['dtype']} | {causal_name(r['causal'])} |"
        if r["status"] != "ok":
            lines.append(f"{prefix} skipped: {r['skip_reason']} |" + " |" * (head.count("|") - 5))
            continue
        ref, sd = (r["methods"][m] for m in ("reference", "sdpa"))
        err = r["correctness"]["max_abs_err"]
        row = (
            f"{prefix} {us(ref['timing']['median_s']):.1f} ({us(ref['timing']['iqr_s']):.1f}) |"
            f" {us(sd['timing']['median_s']):.1f} ({us(sd['timing']['iqr_s']):.1f}) |"
            f" {r['speedup_reference_over_sdpa']:.2f}x | {err['reference']:.1e} / {err['sdpa']:.1e} |"
            f" {'yes' if r['correctness']['passed'] else '**NO**'} |"
        )
        if has_mem:
            row += f" {ref['memory']['incremental_peak_bytes'] / 2**20:.2f} / {sd['memory']['incremental_peak_bytes'] / 2**20:.2f} |"
        lines.append(row)
    return lines


def observations(run: dict) -> list[str]:
    """Statements computed from this run's numbers only."""
    ok = [r for r in run["results"] if r["status"] == "ok"]
    if not ok:
        return ["- No successful configurations in this run; nothing to interpret."]
    faster = sum(r["speedup_reference_over_sdpa"] > 1 for r in ok)
    lines = [f"- SDPA had the lower median latency in {faster} of {len(ok)} configurations."]
    groups = defaultdict(list)
    for r in ok:
        groups[(r["dtype"], r["causal"])].append(r)
    for (dtype, causal), rs in groups.items():
        ratios = [r["speedup_reference_over_sdpa"] for r in rs]
        lines.append(
            f"- {dtype}, {causal_name(causal)}: reference/SDPA ratio {min(ratios):.2f}x to {max(ratios):.2f}x "
            f"over L = {', '.join(str(r['seq_len']) for r in rs)}."
        )
    failed = [r for r in ok if not r["correctness"]["passed"]]
    if failed:
        lines.append(f"- **{len(failed)} configuration(s) failed the correctness check; their timings are suspect.**")
    skipped = [r for r in run["results"] if r["status"] != "ok"]
    if skipped:
        lines.append(f"- {len(skipped)} configuration(s) skipped (reasons in the table).")

    scaling = []
    for (method, dtype, causal), pts in series(run).items():
        if len(pts) >= 2:
            (L0, m0), (L1, m1) = pts[0], pts[-1]
            growth = m1["timing"]["median_s"] / m0["timing"]["median_s"]
            scaling.append(
                f"| {method} | {dtype} | {causal_name(causal)} | {L0} → {L1} | x{growth:.1f} | x{L1 / L0:g} | x{(L1 / L0) ** 2:g} |"
            )
    if scaling:
        lines += [
            "",
            "Latency growth across the measured range, compared with linear and quadratic growth in L:",
            "",
            "| method | dtype | mask | L range | latency growth | L growth | L² growth |",
            "|---|---|---|---|---|---|---|",
            *scaling,
        ]
    return lines


CAVEATS = """\
How to read this:
- One run on one machine. Re-run before quoting numbers; compare only rows from the same environment.
- Timings cover one attention call including the identical Python validation in both wrappers;
  input creation, correctness checks and I/O are excluded. This is not whole-model performance.
- SDPA backend: automatic selection; not verified. Nothing here shows which kernel ran.
- The score matrix is O(L²), but at small L fixed per-call costs (Python, op dispatch, allocation)
  are a large share, so latency can grow much more slowly than L². Growth near L² is only expected
  once the O(L²·D) arithmetic dominates.
- The explicit causal reference builds an [L, L] mask and runs masked_fill on every call, which adds
  work the noncausal path does not do."""


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
        if any(r["status"] == "ok" for r in run["results"]):
            latency_png = out_dir / f"{path.stem}-latency.png"
            plot_latency(run, latency_png)
            md += [f"![latency vs L]({latency_png.name})", ""]
            memory_png = out_dir / f"{path.stem}-memory.png"
            if plot_memory(run, memory_png):
                md += [f"![memory vs L]({memory_png.name})", ""]
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
