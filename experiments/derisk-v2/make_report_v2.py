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

"""Build the derisk-v2 HTML report: derisk-v1 baselines vs v2.1 (random pad tokens) vs v2.2 (expert placement).

Reuses derisk-v1's run loader, routing aggregation and HTML template (``make_report.py`` and
``report_template.html`` in experiments/derisk-v1). Variants whose runs do not exist yet are skipped.

Usage::

    source experiments/derisk-v1/env.sh
    PYTHONPATH=$PYTHONPATH:experiments/derisk-v1 python experiments/derisk-v2/make_report_v2.py \
        --output hub/derisk-v2/index.html
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import make_report as v1report
import numpy as np

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parent / "derisk-v1"
_esc = v1report._esc

VARIANTS = (
    {"key": "v1", "label": "v1 baseline", "color": "--deemph"},
    {"key": "v21", "label": "v2.1 random pad tokens", "color": "--series-1"},
    {"key": "v22", "label": "v2.2 expert placement", "color": "--series-2"},
    {"key": "v23a", "label": "v2.3a natural pad, masked", "color": "--series-3"},
    {"key": "v23b", "label": "v2.3b natural pad, visible", "color": "--series-4"},
)
ARMS = (
    {
        "key": "fixed",
        "label": "pad-fixed8192",
        "desc": "every micro-batch padded to 8,192 tokens, 20 steps",
        "runs": {
            "v1": _V1 / "runs" / "pad_fixed8192",
            "v21": _HERE / "runs" / "v21_pad_fixed8192_randpad",
            "v22": _HERE / "runs" / "v22_pad_fixed8192_placement",
            "v23a": _HERE / "runs" / "v23a_pad_fixed8192_natural_masked",
            "v23b": _HERE / "runs" / "v23b_pad_fixed8192_natural_visible",
        },
        "placement": _HERE / "placements" / "pad_fixed8192.json",
        "windows": (("steps 2–19", 2, 20),),
        "shade": ([0, 1], "warmup"),
    },
    {
        "key": "batchmax",
        "label": "pad-batchmax, two passes",
        "desc": "each micro-batch padded to its longest sequence, 40 steps = two passes over the same data",
        "runs": {
            "v1": _V1 / "runs" / "pad_batchmax_2pass",
            "v21": _HERE / "runs" / "v21_pad_batchmax_randpad_2pass",
            "v22": _HERE / "runs" / "v22_pad_batchmax_placement_2pass",
            "v23a": _HERE / "runs" / "v23a_pad_batchmax_natural_masked_2pass",
            "v23b": _HERE / "runs" / "v23b_pad_batchmax_natural_visible_2pass",
        },
        "placement": _HERE / "placements" / "pad_batchmax_2pass.json",
        "windows": (("pass 1, steps 2–19 (cold shape caches)", 2, 20), ("pass 2, steps 20–39 (warm)", 20, 40)),
        "shade": ([0, 19], "pass 1"),
    },
)
METRICS = (
    ("Step time", "probe/step_time_wall_s", "{:.2f} s", False),
    ("Effective MFU (real tokens)", "probe/mfu_effective_wall", "{:.2f}%", True),
    # No "better" label: padded MFU also rises when more pad work is counted.
    ("Padded MFU (AutoModel mfu)", "probe/mfu_padded_wall", "{:.2f}%", None),
    ("Real tokens / s / GPU", "probe/real_tokens_per_s_per_gpu", "{:,.0f}", True),
    ("EP imbalance, mean", "routing/ep_imbalance_all_mean", "{:.2f}×", False),
    ("EP imbalance, worst dispatch", "routing/ep_imbalance_all_max", "{:.2f}×", False),
    ("Peak reserved memory, max rank", "probe/mem_reserved_peak_gib_max_rank", "{:.1f} GiB", False),
)


def _window_mean(records: list[dict], key: str, start: int, stop: int) -> float:
    values = [r[key] for r in records if start <= r["step"] < stop and r.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _variant_line(title: str, subtitle: str, runs: list[dict], key: str, y: dict, shade: tuple) -> dict:
    return {
        "type": "line",
        "title": title,
        "subtitle": subtitle,
        "series": [
            {"label": r["label"], "color": r["color"], "points": v1report._series(r["records"], key)} for r in runs
        ],
        "x": {"label": "Step", "fmt": "int"},
        "y": y,
        "shade": shade[0],
        "shadeLabel": shade[1],
    }


def build(arms: list[dict]) -> tuple[dict, str]:
    """Compute chart specs and body HTML for every arm with a v1 baseline.

    Args:
        arms: ``ARMS`` entries, each with ``loaded``: variant key -> loaded run (``v1report._load_arm``
            output merged with the variant entry), for the variants whose runs exist.

    Returns:
        The chart payload and the report body HTML.
    """
    charts: dict[str, dict] = {}
    parts: list[str] = []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        '<header class="page"><div><h1>Evening out pad-token expert load (derisk-v2)</h1>'
        f'<div class="meta">Generated {_esc(generated)} · Nemotron 3 Nano SFT on Tulu-3, 8× H100, FSDP2 + EP=8 · '
        'wandb project <code>derisk-v2</code> · baseline: <a href="../derisk-v1/index.html">derisk-v1</a></div></div>'
        '<button type="button" id="theme-toggle">Theme: auto</button></header>'
    )
    parts.append(
        '<p class="lede">derisk-v1 found that every padding token reached the MoE layers with the same hidden state, so the '
        "router sent all of them to the same 6 experts per layer and the GPUs hosting those experts became stragglers. "
        "derisk-v2 tries two ways to even that out, each on the same data, seed and schedule as the derisk-v1 run it is "
        'compared with:</p><ul class="lede">'
        "<li><b>v2.1 random pad tokens</b>: fill padding positions with random ordinary token ids (1000–131071) "
        "instead of <code>&lt;|im_end|&gt;</code>. Masks and labels are unchanged, so the loss on real tokens is unchanged.</li>"
        "<li><b>v2.2 expert placement</b>: keep <code>&lt;|im_end|&gt;</code> padding, but choose per layer which 16 experts "
        "each GPU hosts, from derisk-v1's recorded router counts, so every GPU gets a similar share of each dispatch. "
        "Experts, router rows and router bias are relabeled together, so the model computes the same function.</li>"
        "<li><b>v2.3a natural pad, masked</b>: fill each row's padding with a random window of natural Tulu-3 text "
        "(held-out conversations), still with attention mask 0, so each filler token is routed on its own embedding.</li>"
        "<li><b>v2.3b natural pad, visible</b>: the same filler with attention mask 1 and labels −100, so filler tokens "
        "reach the router with context; causal attention keeps the real tokens from seeing them.</li></ul>"
    )

    findings = []
    for arm in arms:
        base = arm["loaded"].get("v1")
        if base is None:
            continue
        for label, start, stop in arm["windows"]:
            for variant in VARIANTS[1:]:
                run = arm["loaded"].get(variant["key"])
                if run is None:
                    continue
                step = [_window_mean(r["records"], "probe/step_time_wall_s", start, stop) for r in (base, run)]
                mfu = [_window_mean(r["records"], "probe/mfu_effective_wall", start, stop) for r in (base, run)]
                imb = [_window_mean(r["records"], "routing/ep_imbalance_all_mean", start, stop) for r in (base, run)]
                findings.append(
                    f"<b>{_esc(arm['label'])}, {_esc(label)}, {_esc(variant['label'])}</b>: step time {step[0]:.2f} s → "
                    f"{step[1]:.2f} s ({100 * (step[1] - step[0]) / step[0]:+.0f}%), effective MFU {mfu[0]:.2f}% → "
                    f"{mfu[1]:.2f}% ({100 * (mfu[1] - mfu[0]) / mfu[0]:+.0f}%), mean EP imbalance {imb[0]:.2f}× → {imb[1]:.2f}×."
                )
    if findings:
        parts.append(
            '<h2 id="findings">Findings</h2><div class="findings"><ul>'
            + "".join(f"<li>{f}</li>" for f in findings)
            + "</ul></div>"
        )
    notes_path = _HERE / "notes_v2.html"
    if notes_path.exists():
        parts.append(f'<h2 id="interpretation">Interpretation</h2><div class="panel">{notes_path.read_text()}</div>')

    for arm in arms:
        runs = [arm["loaded"][v["key"]] for v in VARIANTS if v["key"] in arm["loaded"]]
        if "v1" not in arm["loaded"] or len(runs) < 2:
            continue
        base = arm["loaded"]["v1"]
        key = arm["key"]
        parts.append(f'<h2 id="{key}">{_esc(arm["label"])}</h2><p class="note">{_esc(arm["desc"])}.</p>')

        # Summary table: one column block per measurement window.
        head = '<tr><th class="text">Metric</th>'
        for label, _, _ in arm["windows"]:
            for run in runs:
                head += f'<th>{_esc(run["label"])}<br><span class="note">{_esc(label)}</span></th>'
        head += "</tr>"
        body = []
        for label, metric, pattern, higher_is_better in METRICS:
            row = f'<tr><td class="text">{_esc(label)}</td>'
            for _, start, stop in arm["windows"]:
                base_value = _window_mean(base["records"], metric, start, stop)
                for run in runs:
                    value = _window_mean(run["records"], metric, start, stop)
                    cell = "–" if np.isnan(value) else pattern.format(value)
                    if run is not base and not np.isnan(value) and base_value:
                        change = 100 * (value - base_value) / base_value
                        good = higher_is_better is not None and (change > 0) == higher_is_better
                        cell += f' <span class="note">({"+" if change >= 0 else ""}{change:.0f}%{" better" if good and abs(change) >= 1 else ""})</span>'
                    row += f"<td>{cell}</td>"
            body.append(row + "</tr>")
        loss_rows = []
        for run in runs:
            if run is base:
                continue
            n = min(len(run["records"]), len(base["records"]))
            diff = max(abs(run["records"][i]["loss"] - base["records"][i]["loss"]) for i in range(n))
            loss_rows.append(f"{_esc(run['label'])}: max |Δloss| vs v1 = {diff:.4f} over {n} steps")
        parts.append(
            f'<div class="panel"><div class="table-wrap"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
            f'<p class="note">Changes are relative to the v1 baseline in the same window. Loss check (same data, so '
            f"the loss should match): {'; '.join(loss_rows)}.</p></div>"
        )

        charts[f"{key}_step"] = _variant_line(
            "Step time",
            "Wall clock between optimizer steps",
            runs,
            "probe/step_time_wall_s",
            {"label": "Seconds", "fmt": "s2", "min": 0, "tickFmt": "s0"},
            arm["shade"],
        )
        charts[f"{key}_mfu"] = _variant_line(
            "Effective MFU",
            "Only real tokens counted, wall clock",
            runs,
            "probe/mfu_effective_wall",
            {"label": "MFU (%)", "fmt": "pct2", "min": 0, "tickFmt": "pct1"},
            arm["shade"],
        )
        charts[f"{key}_imb_step"] = _variant_line(
            "Worst EP imbalance per step",
            "Max over layers and micro-batches of busiest GPU ÷ mean",
            runs,
            "routing/ep_imbalance_all_max",
            {"label": "Imbalance", "fmt": "x2", "min": 1, "tickFmt": "x1"},
            arm["shade"],
        )
        charts[f"{key}_padded"] = _variant_line(
            "Padded MFU",
            "AutoModel's mfu: pads counted as work",
            runs,
            "probe/mfu_padded_wall",
            {"label": "MFU (%)", "fmt": "pct2", "min": 0, "tickFmt": "pct0"},
            arm["shade"],
        )
        parts.append(
            f'<div class="grid2"><div data-chart="{key}_step"></div><div data-chart="{key}_mfu"></div>'
            f'<div data-chart="{key}_imb_step"></div><div data-chart="{key}_padded"></div></div>'
        )

        routed = [r for r in runs if r.get("routing") is not None]
        if routed:
            layer_ids = routed[0]["routing"]["layer_ids"]
            series = [
                {
                    "label": r["label"],
                    "color": r["color"],
                    "points": [[lid, float(v)] for lid, v in zip(layer_ids, r["routing"]["imbalance_layer_mean"])],
                }
                for r in routed
            ]
            charts[f"{key}_imb_layer"] = {
                "type": "line",
                "title": "EP imbalance per layer",
                "subtitle": "Busiest GPU ÷ mean assignments per dispatch, averaged over micro-batches and steps (1.0 = balanced)",
                "series": series,
                "x": {"label": "Layer", "fmt": "int", "ticks": layer_ids[::2]},
                "y": {"label": "Imbalance", "fmt": "x2", "min": 1, "tickFmt": "x1"},
                "markers": True,
                "endLabels": False,
            }
            parts.append(f'<div data-chart="{key}_imb_layer"></div>')
            placement = json.loads(arm["placement"].read_text()) if arm["placement"].exists() else None
            v22 = arm["loaded"].get("v22")
            if placement is not None and v22 is not None and v22.get("routing") is not None:
                predicted = [[lid, placement["layers"][str(lid)]["predicted_imbalance_mean"]] for lid in layer_ids]
                realized = [[lid, float(v)] for lid, v in zip(layer_ids, v22["routing"]["imbalance_layer_mean"])]
                charts[f"{key}_imb_pred"] = {
                    "type": "line",
                    "title": "v2.2: predicted vs realized imbalance per layer",
                    "subtitle": "Predicted on derisk-v1's recorded counts before the run; realized in the v2.2 run",
                    "series": [
                        {"label": "predicted", "color": "--accent", "points": predicted},
                        {"label": "realized", "color": "--series-2", "points": realized},
                    ],
                    "x": {"label": "Layer", "fmt": "int", "ticks": layer_ids[::2]},
                    "y": {"label": "Imbalance", "fmt": "x2", "min": 1, "tickFmt": "x1"},
                    "markers": True,
                }
                parts.append(f'<div data-chart="{key}_imb_pred"></div>')

            # Pad routing (original expert ids: v1 and v2.1 share the labeling) and per-GPU work.
            per_rank = routed[0]["routing"]["n_experts"] // routed[0]["routing"]["ep"]
            heat_runs = [r for r in routed if r["key"] in ("v1", "v21", "v23a", "v23b")]
            if len(heat_runs) >= 2:
                vmax = float(max(r["routing"]["pad_share"].max() for r in heat_runs))
                cells = []
                for r in heat_runs:
                    charts[f"{key}_pad_{r['key']}"] = v1report._heat(
                        f"{r['label']}: pad-token routing",
                        "Share of each layer's pad (filler) assignments per expert; same color scale in every panel",
                        r["routing"]["pad_share"],
                        layer_ids,
                        per_rank,
                        vmax=vmax,
                    )
                    cells.append(f'<div data-chart="{key}_pad_{r["key"]}"></div>')
                parts.append('<div class="grid2">' + "".join(cells) + "</div>")
            gpu_runs = [r for r in routed if r["key"] in ("v1", "v22")]
            if len(gpu_runs) == 2:
                vmax = float(max(r["routing"]["rank_share_all"].max() for r in gpu_runs))
                cells = []
                for r in gpu_runs:
                    matrix = r["routing"]["rank_share_all"]
                    charts[f"{key}_gpu_{r['key']}"] = {
                        "type": "heatmap",
                        "title": f"{r['label']}: share of each layer's expert work per GPU",
                        "subtitle": f"All tokens; balanced = {100 / r['routing']['ep']:.1f}% per GPU (same color scale for both)",
                        "rows": [str(lid) for lid in layer_ids],
                        "cols": [str(g) for g in range(r["routing"]["ep"])],
                        "values": np.round(matrix, 6).tolist(),
                        "vmax": vmax,
                        "fmt": "frac_pct1",
                        "scale": "linear",
                        "rowName": "Layer",
                        "colName": "EP rank",
                        "rowTitle": "MoE layer",
                        "colTitle": "EP rank (GPU)",
                        "valueLabel": "of the layer's assignments",
                    }
                    cells.append(f'<div data-chart="{key}_gpu_{r["key"]}"></div>')
                parts.append('<div class="grid2">' + "".join(cells) + "</div>")

        mem = {
            r["key"]: np.asarray([x["per_rank_mem_reserved_peak_gib"] for x in r["records"]]).max(axis=0) for r in runs
        }
        charts[f"{key}_mem"] = {
            "type": "bars",
            "title": "Peak reserved GPU memory per rank",
            "subtitle": "Max over all steps",
            "categories": [str(i) for i in range(len(next(iter(mem.values()))))],
            "series": [
                {"label": r["label"], "color": r["color"], "values": [float(v) for v in mem[r["key"]]]} for r in runs
            ],
            "x": {"label": "Rank"},
            "y": {"label": "GiB", "fmt": "gib1", "tickFmt": "num0"},
        }
        parts.append(f'<div data-chart="{key}_mem"></div>')
        links = [
            f'{_esc(r["label"])}: <a href="{_esc(r["wandb_url"])}">{_esc(r["wandb_url"])}</a>'
            for r in runs
            if r.get("wandb_url")
        ]
        if links:
            parts.append('<p class="note">wandb: ' + "; ".join(links) + ".</p>")

    parts.append(
        '<h2 id="repro">Reproduce</h2><pre>bash experiments/derisk-v2/run_v2.sh v21_pad_fixed8192_randpad v21_pad_fixed8192_randpad\n'
        "bash experiments/derisk-v2/run_v2.sh v21_pad_batchmax_randpad v21_pad_batchmax_randpad_2pass --step_scheduler.max_steps 40\n"
        "python experiments/derisk-v2/expert_placement.py --routing experiments/derisk-v1/runs/pad_fixed8192/routing.npz "
        "--output experiments/derisk-v2/placements/pad_fixed8192.json\n"
        "bash experiments/derisk-v2/run_v2.sh v22_pad_fixed8192_placement v22_pad_fixed8192_placement\n"
        "bash experiments/derisk-v2/run_v2.sh v22_pad_batchmax_placement v22_pad_batchmax_placement_2pass --step_scheduler.max_steps 40</pre>"
    )
    return {"charts": charts}, "\n".join(parts)


def main() -> None:
    """Load every available run, build the report, and write the self-contained HTML."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    arms = []
    for arm in ARMS:
        loaded = {}
        for variant in VARIANTS:
            run_dir = arm["runs"][variant["key"]]
            run = v1report._load_arm(run_dir) if (run_dir / "exit.json").exists() else None
            if run is not None:
                loaded[variant["key"]] = {**variant, **run}
        arms.append({**arm, "loaded": loaded})
    payload, body = build(arms)
    template = (_V1 / "report_template.html").read_text()
    page = (
        template.replace("{{TITLE}}", "derisk-v2: evening out pad-token expert load")
        .replace("{{BODY_HTML}}", body)
        .replace("{{PAYLOAD_JSON}}", json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    logger.info("wrote %s (%.0f KiB)", args.output, len(page) / 1024)


if __name__ == "__main__":
    main()
