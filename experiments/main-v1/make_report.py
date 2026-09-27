# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the main-v1 HTML report from the run directories (CPU only).

Every number in the report is computed here from ``runs/<name>/probe_steps.jsonl`` and ``routing.npz`` and
``data/data_stats.json``; the interpretation text is ``notes.html``, inserted verbatim.

Usage::

    source experiments/main-v1/env.sh
    python experiments/main-v1/make_report.py                 # writes hub/main-v1/index.html
    python experiments/main-v1/make_report.py --print-only    # prints the key numbers instead
"""

from __future__ import annotations

import argparse
import csv
import datetime
import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
WORLD = 8
PEAK_TFLOPS = 989.0
EXPERTS_PER_GPU = 16
# nvidia-smi clocks_event_reasons bits: 0x20 software thermal slowdown, 0x40 hardware thermal slowdown.
THERMAL_BITS = 0x20 | 0x40
_STEP_LOG_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| INFO \| root \| step (\d+) \|")
MIN_STEPS = 10  # a run enters a window's averages and comparisons only with at least this many clean steps
MICROBATCHES_PER_STEP = 8  # per GPU: 128 conversations / (8 GPUs * 2 per micro-batch)


@dataclass(frozen=True)
class Method:
    """One padding method: its config suffix, display name, chart color and dash style."""

    key: str
    label: str
    color: str
    dash: bool = False


METHODS = (
    Method("default", "Default", "--deemph"),
    Method("random", "Random tokens", "--series-1"),
    Method("placement", "Expert placement", "--series-2"),
    Method("natural_masked", "Natural text, hidden", "--series-3"),
    Method("natural_visible", "Natural text, visible", "--series-4"),
)
REPEAT = Method("default_repeat", "Default, second run", "--deemph", dash=True)
SETTINGS = {"fixed": "Fixed length", "batchmax": "Longest in batch"}


@dataclass(frozen=True)
class Window:
    """Steps [start, stop) of one setting that are averaged together."""

    key: str
    setting: str
    label: str
    start: int
    stop: int


WINDOWS = (
    Window("fixed", "fixed", "Fixed length", 2, 40),
    Window("batchmax_pass1", "batchmax", "Longest in batch, first pass", 2, 40),
    Window("batchmax_pass2", "batchmax", "Longest in batch, second pass", 42, 80),
)


@dataclass
class Run:
    """Everything the report reads from one run directory."""

    name: str
    steps: dict[int, dict]
    counts: np.ndarray
    """Router assignment counts, [recorded steps, micro-batches, MoE layers, token categories, experts]."""
    routing_steps: list[int]
    layer_ids: list[int]
    microbatch_shapes: np.ndarray
    """Per step, GPU and micro-batch: (batch, padded length, real tokens), [recorded steps, GPUs, micro-batches, 3]."""
    overheated_steps: set[int]
    """Steps during which any GPU reported thermal slowdown; left out of every average and comparison."""
    wandb_url: str | None
    start_utc: str
    end_utc: str


def load_run(runs_dir: Path, name: str) -> Run | None:
    """Load one finished run, or return ``None`` if it has not finished successfully."""
    run_dir = runs_dir / name
    exit_path = run_dir / "exit.json"
    if not exit_path.exists() or json.loads(exit_path.read_text())["exit_code"] != 0:
        return None
    records = [json.loads(line) for line in (run_dir / "probe_steps.jsonl").read_text().splitlines()]
    routing = np.load(run_dir / "routing.npz", allow_pickle=False)
    match = re.search(r"https://wandb\.ai/\S+/runs/\w+", (run_dir / "train.log").read_text(errors="replace"))
    return Run(
        name=name,
        steps={int(record["step"]): record for record in records},
        counts=routing["counts"],
        routing_steps=routing["steps"].tolist(),
        layer_ids=routing["layer_ids"].tolist(),
        microbatch_shapes=routing["microbatch_shapes"],
        overheated_steps=_overheated_steps(run_dir),
        wandb_url=match.group(0) if match else None,
        start_utc=json.loads((run_dir / "launch.json").read_text())["start_utc"],
        end_utc=json.loads(exit_path.read_text())["end_utc"],
    )


def _overheated_steps(run_dir: Path) -> set[int]:
    """Steps that overlap a GPU thermal slowdown sample in ``gpu_telemetry.csv`` (500 ms nvidia-smi samples).

    A step spans from the previous step's log line to its own (``train.log``, 1 s resolution), padded by 1 s.
    """
    hot = []
    with (run_dir / "gpu_telemetry.csv").open() as f:
        reader = csv.reader(f)
        header = [name.strip() for name in next(reader)]
        time_col, reason_col = header.index("timestamp"), header.index("clocks_event_reasons.active")
        for row in reader:
            if len(row) == len(header) and int(row[reason_col].strip(), 16) & THERMAL_BITS:
                hot.append(datetime.datetime.strptime(row[time_col].strip()[:19], "%Y/%m/%d %H:%M:%S"))
    if not hot:
        return set()
    log = (run_dir / "train.log").read_text(errors="replace")
    ends = {int(step): datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t, step in _STEP_LOG_RE.findall(log)}
    margin = datetime.timedelta(seconds=1)
    overheated, previous = set(), None
    for step in sorted(ends):
        start = (previous if previous is not None else ends[step] - datetime.timedelta(minutes=5)) - margin
        if any(start <= t <= ends[step] + margin for t in hot):
            overheated.add(step)
        previous = ends[step]
    return overheated


def _window_steps(run: Run, window: Window) -> list[int]:
    """Steps of the window that the run recorded and that no GPU spent overheating."""
    return [s for s in range(window.start, window.stop) if s in run.steps and s not in run.overheated_steps]


def _gpu_loads(run: Run, window: Window) -> np.ndarray:
    """Expert assignments each GPU received: array of shape [dispatches, MoE layers, GPUs] for the window."""
    clean = set(_window_steps(run, window))
    rows = [i for i, step in enumerate(run.routing_steps) if step in clean]
    counts = run.counts[rows].sum(axis=3)  # [steps, micro-batches, layers, experts], all token categories
    n_steps, n_micro, n_layers, n_experts = counts.shape
    loads = counts.reshape(n_steps, n_micro, n_layers, n_experts // EXPERTS_PER_GPU, EXPERTS_PER_GPU).sum(axis=-1)
    return loads.reshape(n_steps * n_micro, n_layers, -1).astype(np.float64)


def window_metrics(run: Run, window: Window) -> dict[str, float]:
    """Averages of one run over one window (ratios of sums, so long steps weigh by their length)."""
    records = [run.steps[step] for step in _window_steps(run, window)]
    wall = np.asarray([r["probe/step_time_wall_s"] for r in records])
    useful_tflop = np.asarray([r["probe/tflops_per_gpu_effective_wall"] for r in records]) * WORLD * wall
    padded_tflop = np.asarray([r["probe/tflops_per_gpu_padded_wall"] for r in records]) * WORLD * wall
    real = sum(r["probe/tokens_real"] for r in records)
    padded = sum(r["probe/tokens_padded"] for r in records)
    loads = _gpu_loads(run, window)
    imbalance = loads.max(axis=-1) / np.maximum(loads.mean(axis=-1), 1e-9)
    clean = set(_window_steps(run, window))
    rows = [i for i, step in enumerate(run.routing_steps) if step in clean]
    # Per dispatch (micro-batch) and layer: share of padding assignments that go to the 6 most used experts.
    pad_counts = run.counts[rows][:, :, :, 1:, :].sum(axis=3).astype(np.float64)  # [steps, micro, layers, experts]
    real_counts = run.counts[rows][:, :, :, 0, :].astype(np.float64)
    pad_top6 = np.sort(pad_counts, axis=-1)[..., ::-1][..., :6].sum(axis=-1) / np.maximum(pad_counts.sum(-1), 1)
    real_top6 = np.sort(real_counts, axis=-1)[..., ::-1][..., :6].sum(axis=-1) / np.maximum(real_counts.sum(-1), 1)
    return {
        "steps": len(records),
        "step_time_s": float(wall.mean()),
        "useful_mfu_pct": float(100 * useful_tflop.sum() / (wall.sum() * WORLD * PEAK_TFLOPS)),
        "padded_mfu_pct": float(100 * padded_tflop.sum() / (wall.sum() * WORLD * PEAK_TFLOPS)),
        "pad_fraction": float(1.0 - real / padded),
        "real_tokens_per_s_per_gpu": float(real / (wall.sum() * WORLD)),
        "tokens_per_microbatch_per_gpu": float(padded / (len(records) * WORLD * MICROBATCHES_PER_STEP)),
        "imbalance_mean": float(imbalance.mean()),
        "imbalance_worst": float(imbalance.max()),
        "peak_mem_gib": float(max(r["probe/mem_reserved_peak_gib_max_rank"] for r in records)),
        "pad_top6_share": float(pad_top6.mean()),
        "real_top6_share": float(real_top6.mean()),
    }


def new_length_share(run: Run, window: Window) -> float:
    """Share of the window's micro-batches whose padded length their GPU had not processed earlier in the run."""
    seen: list[set[int]] = [set() for _ in range(run.microbatch_shapes.shape[1])]
    new = total = 0
    for row, step in enumerate(run.routing_steps):
        for gpu, lengths in enumerate(run.microbatch_shapes[row, :, :, 1].tolist()):
            for length in lengths:
                if window.start <= step < window.stop:
                    new += length not in seen[gpu]
                    total += 1
                seen[gpu].add(length)
    return new / max(total, 1)


def gpu_work_share(run: Run, window: Window) -> np.ndarray:
    """Share of each MoE layer's expert work done by each GPU over the window: array [MoE layers, GPUs]."""
    total = _gpu_loads(run, window).sum(axis=0)
    return total / total.sum(axis=1, keepdims=True)


def paired_change(run: Run, baseline: Run, window: Window) -> dict[str, float]:
    """Step-by-step step-time difference ``run - baseline`` over the window, with a 95% t-interval.

    Both runs see the same micro-batches at every step, so the per-step differences cancel most of the
    step-to-step variation that comes from the data.
    """
    steps = [step for step in _window_steps(run, window) if step in set(_window_steps(baseline, window))]
    run_s = np.asarray([run.steps[step]["probe/step_time_wall_s"] for step in steps])
    base_s = np.asarray([baseline.steps[step]["probe/step_time_wall_s"] for step in steps])
    diff = run_s - base_s
    half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    base = base_s.mean()
    return {
        "steps": len(steps),
        "run_s": float(run_s.mean()),
        "base_s": float(base),
        "delta_s": float(diff.mean()),
        "delta_lo_s": float(diff.mean() - half),
        "delta_hi_s": float(diff.mean() + half),
        "pct": float(100 * diff.mean() / base),
        "pct_lo": float(100 * (diff.mean() - half) / base),
        "pct_hi": float(100 * (diff.mean() + half) / base),
    }


def max_loss_gap(run: Run, baseline: Run) -> float:
    """Largest absolute per-step loss difference between two runs over all common steps."""
    steps = sorted(run.steps.keys() & baseline.steps.keys())
    return float(max(abs(run.steps[s]["loss"] - baseline.steps[s]["loss"]) for s in steps))


def load_runs(runs_dir: Path) -> dict[str, Run]:
    """Load every finished measured run, keyed by run name."""
    runs = {}
    for setting in SETTINGS:
        for method in (*METHODS, REPEAT):
            run = load_run(runs_dir, f"{setting}_{method.key}")
            if run is not None:
                runs[run.name] = run
    return runs


def analyze(runs: dict[str, Run]) -> dict:
    """Compute every number the report shows, keyed by window and method."""
    results: dict = {"windows": {}, "loss_gap": {}}
    for window in WINDOWS:
        baseline = runs.get(f"{window.setting}_default")
        rows = {}
        for method in (*METHODS, REPEAT):
            run = runs.get(f"{window.setting}_{method.key}")
            if run is None or len(_window_steps(run, window)) < MIN_STEPS:
                continue
            row = window_metrics(run, window)
            if baseline is not None and method.key != "default" and len(_window_steps(baseline, window)) >= MIN_STEPS:
                row["change"] = paired_change(run, baseline, window)
            rows[method.key] = row
        results["windows"][window.key] = rows
    for setting in SETTINGS:
        baseline = runs.get(f"{setting}_default")
        for method in (*METHODS[1:], REPEAT):
            run = runs.get(f"{setting}_{method.key}")
            if run is not None and baseline is not None:
                results["loss_gap"][f"{setting}_{method.key}"] = max_loss_gap(run, baseline)
    return results


def print_results(results: dict) -> None:
    """Log the key numbers of every window."""
    for window in WINDOWS:
        rows = results["windows"][window.key]
        logger.info("== %s (steps %d-%d)", window.label, window.start, window.stop - 1)
        for key, row in rows.items():
            change = row.get("change")
            change_text = (
                f"{change['pct']:+.2f}% [{change['pct_lo']:+.2f}, {change['pct_hi']:+.2f}]" if change else "baseline"
            )
            logger.info(
                "  %-16s %6.2f s  %s  useful MFU %5.2f%%  padded MFU %5.2f%%  pad %4.1f%%  busiest GPU %.2fx (worst %.2fx)"
                "  mem %.1f GiB",
                key,
                row["step_time_s"],
                change_text,
                row["useful_mfu_pct"],
                row["padded_mfu_pct"],
                100 * row["pad_fraction"],
                row["imbalance_mean"],
                row["imbalance_worst"],
                row["peak_mem_gib"],
            )
    logger.info("largest per-step loss difference vs default: %s", json.dumps(results["loss_gap"]))


# ---------------------------------------------------------------------------------------------------------
# HTML


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}%".replace("-", "−")


def _method(key: str) -> Method:
    return next(m for m in (*METHODS, REPEAT) if m.key == key)


def effect_chart(results: dict) -> dict:
    """Dot-and-interval chart of every method's step-time change against the default run."""
    groups, table_rows = [], []
    for window in WINDOWS:
        rows = results["windows"][window.key]
        group_rows = []
        for method in (*METHODS[1:], REPEAT):
            row = rows.get(method.key)
            if row is None or "change" not in row:
                continue
            c = row["change"]
            group_rows.append(
                {
                    "label": method.label,
                    "color": method.color,
                    "mean": c["pct"],
                    "lo": c["pct_lo"],
                    "hi": c["pct_hi"],
                    "valueLabel": _fmt_pct(c["pct"]),
                    "valueLabel2": f"{c['base_s']:.2f} → {c['run_s']:.2f} s",
                    "tipHead": f"{window.label} · {method.label}",
                    "tipRows": [
                        {"value": _fmt_pct(c["pct"], 2), "label": "mean change in step time"},
                        {"value": f"{_fmt_pct(c['pct_lo'], 2)} to {_fmt_pct(c['pct_hi'], 2)}", "label": "95% interval"},
                        {"value": f"{c['base_s']:.2f} → {c['run_s']:.2f} s", "label": "default → this method"},
                        {"value": str(c["steps"]), "label": "paired steps"},
                    ],
                }
            )
            table_rows.append(
                [
                    window.label,
                    method.label,
                    f"{c['base_s']:.2f}",
                    f"{c['run_s']:.2f}",
                    _fmt_pct(c["pct"], 2),
                    f"{_fmt_pct(c['pct_lo'], 2)} to {_fmt_pct(c['pct_hi'], 2)}",
                    str(c["steps"]),
                ]
            )
        if group_rows:
            groups.append({"label": f"{window.label} (steps {window.start + 1}–{window.stop})", "rows": group_rows})
    return {
        "type": "forest",
        "groups": groups,
        "x": {
            "label": "Change in step time vs the default method",
            "fmt": "pct0",
            "leftLabel": "← faster",
            "rightLabel": "slower →",
        },
        "valueTitles": ["Change", "Default → method"],
        "tableColumns": ["Setting", "Method", "Default (s)", "Method (s)", "Change", "95% interval", "Steps"],
        "tableRows": table_rows,
    }


def step_time_chart(runs: dict[str, Run], setting: str) -> dict:
    """Step time at every step of every run of one setting."""
    series = []
    for method in (*METHODS, REPEAT):
        run = runs.get(f"{setting}_{method.key}")
        if run is None:
            continue
        points = [[step + 1, run.steps[step]["probe/step_time_wall_s"]] for step in sorted(run.steps) if step >= 1]
        series.append({"label": method.label, "color": method.color, "dash": method.dash, "points": points})
    spec = {
        "type": "line",
        "series": series,
        "x": {"label": "Step", "fmt": "int"},
        "y": {"label": "Step time (s)", "fmt": "s2", "tickFmt": "s0", "min": 0},
        "height": 260,
        "endLabels": False,
    }
    if setting == "batchmax" and series:
        spec["shade"] = [2, 40]
        spec["shadeLabel"] = "first pass"
    return spec


def bar_chart(results: dict, metric: str, windows: tuple[Window, ...], y: dict, ref: dict | None = None) -> dict:
    """Grouped bars: one group per window, one bar per method."""
    series = []
    for method in METHODS:
        values = [results["windows"][w.key].get(method.key, {}).get(metric) for w in windows]
        if any(v is not None for v in values):
            series.append({"label": method.label, "color": method.color, "values": values})
    spec = {"type": "bars", "categories": [w.label for w in windows], "series": series, "x": {"label": ""}, "y": y}
    if ref:
        spec["refLine"] = ref
    return spec


def heatmap_chart(runs: dict[str, Run], name: str, window: Window, vmax: float) -> dict | None:
    """Share of each MoE layer's expert work done by each GPU, for one run."""
    run = runs.get(name)
    if run is None:
        return None
    share = gpu_work_share(run, window)
    return {
        "type": "heatmap",
        "rows": [str(layer) for layer in run.layer_ids],
        "cols": [str(gpu) for gpu in range(share.shape[1])],
        "values": share.tolist(),
        "vmax": vmax,
        "fmt": "frac_pct1",
        "rowName": "Layer",
        "colName": "GPU",
        "rowTitle": "MoE layer",
        "colTitle": "GPU",
        "valueLabel": "of this layer's expert work",
        "cellHeight": 11,
    }


def results_table(results: dict) -> str:
    """HTML table with every measured number per window and method."""
    head = (
        "<tr><th>Setting</th><th>Method</th><th class='num'>Step time</th><th class='num'>Change vs default"
        "</th><th class='num'>Useful MFU</th><th class='num'>Padded MFU</th><th class='num'>Real tokens/s per GPU"
        "</th><th class='num'>Busiest GPU's expert work</th><th class='num'>Peak memory</th></tr>"
    )
    body = []
    for window in WINDOWS:
        for key, row in results["windows"][window.key].items():
            change = row.get("change")
            change_text = (
                f"{_fmt_pct(change['pct'])} <span class='note'>({_fmt_pct(change['pct_lo'])} to "
                f"{_fmt_pct(change['pct_hi'])})</span>"
                if change
                else "—"
            )
            body.append(
                f"<tr><td>{html.escape(window.label)}</td><td>{html.escape(_method(key).label)}</td>"
                f"<td class='num'>{row['step_time_s']:.2f} s</td><td class='num'>{change_text}</td>"
                f"<td class='num'>{row['useful_mfu_pct']:.2f}%</td><td class='num'>{row['padded_mfu_pct']:.2f}%</td>"
                f"<td class='num'>{row['real_tokens_per_s_per_gpu']:,.0f}</td>"
                f"<td class='num'>{row['imbalance_mean']:.2f}× <span class='note'>(worst {row['imbalance_worst']:.2f}×)"
                f"</span></td><td class='num'>{row['peak_mem_gib']:.1f} GiB</td></tr>"
            )
    return f"<div class='panel table-wrap' style='max-height:none'><table class='plain'>{head}{''.join(body)}</table></div>"


def runs_table(runs: dict[str, Run]) -> str:
    """HTML table listing every run in launch order with its wandb link."""
    rows = []
    for run in sorted(runs.values(), key=lambda r: r.start_utc):
        link = f"<a href='{html.escape(run.wandb_url)}'>wandb</a>" if run.wandb_url else ""
        rows.append(
            f"<tr><td><code>{run.name}</code></td><td>{run.start_utc}</td><td>{run.end_utc}</td>"
            f"<td class='num'>{len(run.steps)}</td><td class='num'>{len(run.overheated_steps)}</td><td>{link}</td></tr>"
        )
    return (
        "<table class='plain'><tr><th>Run</th><th>Start (UTC)</th><th>End (UTC)</th><th class='num'>Steps</th>"
        f"<th class='num'>Steps left out (a GPU overheated)</th><th>Metrics</th></tr>{''.join(rows)}</table>"
    )


def text_values(runs: dict[str, Run], results: dict, data_stats: dict) -> dict[str, str]:
    """Formatted numbers that ``notes.html`` can reference as ``{{key}}``."""
    values: dict[str, str] = {}
    for window in WINDOWS:
        for key, row in results["windows"][window.key].items():
            prefix = f"{window.key}.{key}"
            values[f"{prefix}.step"] = f"{row['step_time_s']:.2f} s"
            values[f"{prefix}.umfu"] = f"{row['useful_mfu_pct']:.2f}%"
            values[f"{prefix}.pmfu"] = f"{row['padded_mfu_pct']:.1f}%"
            values[f"{prefix}.imb"] = f"{row['imbalance_mean']:.2f}×"
            values[f"{prefix}.imbworst"] = f"{row['imbalance_worst']:.2f}×"
            values[f"{prefix}.mem"] = f"{row['peak_mem_gib']:.1f} GiB"
            values[f"{prefix}.pad"] = f"{100 * row['pad_fraction']:.0f}%"
            values[f"{prefix}.tps"] = f"{row['real_tokens_per_s_per_gpu']:,.0f}"
            values[f"{prefix}.mbtokens"] = f"{row['tokens_per_microbatch_per_gpu']:,.0f}"
            values[f"{prefix}.top6"] = f"{100 * row['pad_top6_share']:.0f}%"
            values[f"{prefix}.realtop6"] = f"{100 * row['real_top6_share']:.0f}%"
            change = row.get("change")
            if change:
                values[f"{prefix}.pct"] = _fmt_pct(change["pct"])
                values[f"{prefix}.abspct"] = f"{abs(change['pct']):.1f}%"
                values[f"{prefix}.ci"] = f"{_fmt_pct(change['pct_lo'])} to {_fmt_pct(change['pct_hi'])}"
                values[f"{prefix}.pct2"] = _fmt_pct(change["pct"], 2)
                values[f"{prefix}.ci2"] = f"{_fmt_pct(change['pct_lo'], 2)} to {_fmt_pct(change['pct_hi'], 2)}"
                values[f"{prefix}.delta"] = f"{change['delta_s']:+.2f} s".replace("-", "−")
    for method in METHODS:
        run = runs.get(f"fixed_{method.key}")
        if run is not None:
            share = gpu_work_share(run, WINDOWS[0])
            values[f"heat.{method.key}.maxshare"] = f"{100 * np.median(share.max(axis=1)):.0f}%"
            values[f"heat.{method.key}.maxsharetop"] = f"{100 * share.max():.0f}%"
            values[f"heat.{method.key}.minshare"] = f"{100 * np.median(share.min(axis=1)):.0f}%"
    if "batchmax_default" in runs:
        for window in WINDOWS[1:]:
            share = new_length_share(runs["batchmax_default"], window)
            values[f"{window.key}.newlen"] = f"{100 * share:.0f}%"
    for key, gap in results["loss_gap"].items():
        values[f"loss.{key}"] = f"{gap:.4f}"
    if results["loss_gap"]:
        values["loss.max"] = f"{max(results['loss_gap'].values()):.4f}"
    losses = [r["loss"] for key, run in runs.items() if key.endswith("_default") for r in run.steps.values()]
    if losses:
        values["loss.min"] = f"{min(losses):.2f}"
        values["loss.maxvalue"] = f"{max(losses):.2f}"
    train = data_stats["train"]
    values.update(
        {
            "data.source_rows": f"{data_stats['source']['rows']:,}",
            "data.rows": f"{train['rows']:,}",
            "data.mean_tokens": f"{train['tokens']['mean']:,.0f}",
            "data.median_tokens": f"{train['tokens']['p50']:,.0f}",
            "data.p90_untruncated": f"{train['tokens_untruncated']['p90']:,.0f}",
            "data.truncated_pct": f"{100 * train['truncated_share']:.0f}%",
            "data.reasoning_off_pct": f"{100 * train['reasoning']['off'] / train['rows']:.0f}%",
            "data.turns_mean": f"{train['turns_user']['mean']:.1f}",
            "data.label_share": f"{100 * train['label_tokens']['mean'] / train['tokens']['mean']:.0f}%",
            "data.filler_rows": f"{data_stats['filler']['rows']:,}",
            "data.filler_tokens": f"{data_stats['filler']['tokens'] / 1e6:.1f} million",
            "date": max((run.end_utc for run in runs.values()), default="")[:10],
        }
    )
    return values


def build_report(runs: dict[str, Run], results: dict, data_stats: dict, notes: str) -> tuple[str, dict]:
    """Return the report body (``notes.html`` with its ``{{key}}`` placeholders filled) and the chart payload.

    Raises:
        KeyError: If ``notes.html`` references a value that the runs do not provide.
    """
    charts = {"effect": effect_chart(results)}
    charts["mfu"] = bar_chart(
        results, "useful_mfu_pct", WINDOWS, {"label": "Useful MFU", "fmt": "pct2", "tickFmt": "pct0"}
    )
    charts["imbalance"] = bar_chart(
        results,
        "imbalance_mean",
        WINDOWS,
        {"label": "Busiest GPU's expert work ÷ average", "fmt": "x2", "tickFmt": "x1"},
        ref={"value": 1.0, "label": "perfectly even"},
    )
    for setting in SETTINGS:
        charts[f"steps_{setting}"] = step_time_chart(runs, setting)
    fixed = WINDOWS[0]
    shares = {m.key: gpu_work_share(runs[f"fixed_{m.key}"], fixed) for m in METHODS if f"fixed_{m.key}" in runs}
    vmax = float(np.ceil(max((share.max() for share in shares.values()), default=0.5) * 20) / 20)
    for key in shares:
        charts[f"work_{key}"] = heatmap_chart(runs, f"fixed_{key}", fixed, vmax)
    titles = {
        "effect": (
            "Figure 1. Change in step time for each method",
            "Compared with the default method at the same steps (same data)",
        ),
        "mfu": ("Figure 2. Useful MFU by method", "Share of the 8 GPUs' peak BF16 throughput spent on real tokens"),
        "imbalance": (
            "Figure 3. How uneven the expert work is",
            "Expert work on the busiest GPU ÷ the average GPU, averaged over MoE layers and micro-batches",
        ),
        "steps_fixed": ("Figure 5a. Step time at every step: fixed length", "Step 1 (about 40 s of setup) not shown"),
        "steps_batchmax": (
            "Figure 5b. Step time at every step: longest in batch",
            "Steps 1–40: first pass over the data; steps 41–80: second pass over the same data",
        ),
    }
    for method in METHODS:
        titles[f"work_{method.key}"] = (f"Figure 4{'abcde'[METHODS.index(method)]}. {method.label}", "")
    for key, spec in charts.items():
        spec["title"], spec["subtitle"] = titles[key]
    values = text_values(runs, results, data_stats)
    values["RESULTS_TABLE"] = results_table(results)
    values["RUNS_TABLE"] = runs_table(runs)
    body = re.sub(r"\{\{([\w.]+)\}\}", lambda match: values[match.group(1)], notes)
    return body, {"charts": charts}


def main() -> None:
    """Load the runs, compute the numbers, and write the report (or print the numbers)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=_HERE / "runs")
    parser.add_argument("--output", type=Path, default=_REPO / "hub" / "main-v1" / "index.html")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs = load_runs(args.runs_dir)
    results = analyze(runs)
    print_results(results)
    if args.print_only:
        return
    data_stats = json.loads((_HERE / "data" / "data_stats.json").read_text())
    notes = (_HERE / "notes.html").read_text()
    body, payload = build_report(runs, results, data_stats, notes)
    template = (_HERE / "report_template.html").read_text()
    page = (
        template.replace("{{TITLE}}", "Padding and MoE training speed — main-v1")
        .replace("{{BODY_HTML}}", body)
        .replace("{{PAYLOAD_JSON}}", json.dumps(payload).replace("</", "<\\/"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    logger.info("wrote %s (%.0f KiB) at %s", args.output, len(page) / 1024, datetime.datetime.now().isoformat())


if __name__ == "__main__":
    main()
