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

"""Build the choose-datasets-v1 HTML report from the phase-1 screen (``runs/screen.json``).

Reuses derisk-v1's self-contained HTML template (``report_template.html``; charts render from a JSON payload).
The interpretation text is ``notes.html`` in this directory.

Usage::

    source experiments/derisk-v1/env.sh
    python experiments/choose-datasets-v1/make_report.py --output hub/choose-datasets-v1/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_TEMPLATE = _HERE.parent / "derisk-v1" / "report_template.html"
CAPS = ("4096", "8192")
DATASETS = (
    {
        "key": "tulu3",
        "label": "Tulu-3 (baseline)",
        "color": "--deemph",
        "repo": "allenai/tulu-3-sft-mixture",
        "license": "ODC-BY",
        "note": "derisk-v1/v2's data: seed-42 shuffle of the train split (first 6,000 rows; the first 2,560 are "
        "exactly derisk's subset)",
    },
    {
        "key": "nemotron_ifchat_v1",
        "label": "Nemotron IF-Chat-v1",
        "color": "--series-1",
        "repo": "nvidia/Nemotron-Instruction-Following-Chat-v1 · chat_if",
        "license": "ODC-BY",
        "note": "Nemotron 3 Nano's own SFT data (used_in: nano_v3); sampled from the Hub parquet export, which "
        "holds the first 315,457 of 426,009 rows",
    },
    {
        "key": "mixture_of_thoughts",
        "label": "Mixture-of-Thoughts",
        "color": "--series-2",
        "repo": "open-r1/Mixture-of-Thoughts · all",
        "license": "Apache-2.0 / CC-BY-4.0 sources",
        "note": "DeepSeek-R1 reasoning traces for math, code and science (OpenR1-Distill-7B's SFT data)",
    },
    {
        "key": "wildchat_1m",
        "label": "WildChat-1M",
        "color": "--series-3",
        "repo": "allenai/WildChat-1M",
        "license": "ODC-BY",
        "note": "real user conversations with GPT-3.5 / GPT-4, 57% English",
    },
)


def _esc(value: object) -> str:
    return html.escape(str(value))


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def _range(stat: dict, fmt: str, unit: str, scale: float = 1.0) -> str:
    low, high = format(scale * stat["p5"], fmt), format(scale * stat["p95"], fmt)
    return f'{format(scale * stat["mean"], fmt)}{unit}<br><span class="note" style="white-space:nowrap">{low}–{high}</span>'


def _table(head: list[str], rows: list[list[str]], caption: str = "") -> str:
    cap = f'<p class="note">{caption}</p>' if caption else ""
    header = "".join(f'<th class="{"text" if i == 0 else ""}">{h}</th>' for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"text" if i == 0 else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f'<div class="panel"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>{cap}</div>'


def _charts(screen: dict, lengths: dict[str, dict[str, np.ndarray]]) -> dict[str, dict]:
    ds = [d for d in DATASETS if d["key"] in screen["datasets"]]

    def bars(title: str, subtitle: str, x_label: str, categories: list[str], values, y: dict) -> dict:
        return {
            "type": "bars",
            "title": title,
            "subtitle": subtitle,
            "categories": categories,
            "series": [{"label": d["label"], "color": d["color"], "values": values(d["key"])} for d in ds],
            "x": {"label": x_label},
            "y": y,
            "height": 250,
        }

    def screened(key: str, cap: str, lbs: str = "lbs2") -> dict:
        return screen["datasets"][key]["caps"][cap]["screen"][lbs]

    grid = np.arange(0, 16384 + 1, 128)
    charts = {
        "saving": bars(
            "Predicted step-time saving from random pad tokens",
            "2 sequences per GPU, pad to the longest in each micro-batch; mean over 20 random 2,560-sample orders",
            "Truncation",
            ["4,096", "8,192"],
            lambda k: [100 * screened(k, c)["saving_frac"]["mean"] for c in CAPS],
            {"label": "Step time saved", "fmt": "pct1", "tickFmt": "pct0"},
        ),
        "cdf": {
            "type": "line",
            "title": "Token length per sample (untruncated)",
            "subtitle": "Share of samples at or below each length, Nemotron 3 Nano tokenizer and chat template",
            "series": [
                {
                    "label": d["label"],
                    "color": d["color"],
                    "points": [[int(x), float(100 * np.mean(lengths[d["key"]]["tokens_full"] <= x))] for x in grid],
                }
                for d in ds
            ],
            "x": {"label": "Tokens", "fmt": "compact"},
            "y": {"label": "Samples at or below", "fmt": "pct1", "tickFmt": "pct0", "min": 0, "max": 100},
            "shade": [8192, 16384],
            "shadeLabel": "cut at 8,192",
            "height": 280,
        },
        "padshare": bars(
            "Padding share of the tokens",
            "2 sequences per GPU",
            "Truncation",
            ["4,096", "8,192"],
            lambda k: [100 * screened(k, c)["pad_fraction"]["mean"] for c in CAPS],
            {"label": "Pad tokens / all tokens", "fmt": "pct1", "tickFmt": "pct0"},
        ),
        "tokens": bars(
            "Tokens per micro-batch per GPU",
            "2 sequences per GPU, padding included",
            "Truncation",
            ["4,096", "8,192"],
            lambda k: [screened(k, c)["tokens_per_microbatch"]["mean"] for c in CAPS],
            {"label": "Tokens", "fmt": "compact", "tickFmt": "compact"},
        ),
        "imbalance": bars(
            "Predicted EP imbalance",
            "Busiest GPU ÷ mean, averaged over dispatches and MoE layers; truncation 8,192, 2 sequences per GPU",
            "Pad token",
            ["<|im_end|>", "random ids"],
            lambda k: [screened(k, "8192")["imbalance_eos"]["mean"], screened(k, "8192")["imbalance_random"]["mean"]],
            {"label": "Busiest ÷ mean", "fmt": "x2", "tickFmt": "x1"},
        ),
        "lbs": bars(
            "Predicted saving with 2 vs 4 sequences per GPU",
            "4 per GPU at 8,192 allows 32,768-token micro-batches, which ran out of memory in derisk-v1",
            "Sequences per GPU · truncation",
            ["2 · 4,096", "4 · 4,096", "2 · 8,192", "4 · 8,192"],
            lambda k: [100 * screened(k, c, f"lbs{b}")["saving_frac"]["mean"] for c in CAPS for b in (2, 4)],
            {"label": "Step time saved", "fmt": "pct1", "tickFmt": "pct0"},
        ),
    }
    return charts


def build(screen: dict, lengths: dict[str, dict[str, np.ndarray]], metas: dict[str, dict]) -> tuple[dict, str]:
    """Return the chart payload and the body HTML."""
    ds = [d for d in DATASETS if d["key"] in screen["datasets"]]
    res = screen["datasets"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        '<header class="page"><div><h1>Choosing a dataset where random pad tokens pay off (choose-datasets-v1)</h1>'
        f'<div class="meta">Generated {_esc(generated)} · phase 1: CPU screen, no GPU runs · Nemotron 3 Nano 30B-A3B '
        "SFT, 8× H100, FSDP2 + EP=8 · follows "
        '<a href="../derisk-v2/index.html">derisk-v2</a></div></div>'
        '<button type="button" id="theme-toggle">Theme: auto</button></header>',
        '<p class="lede">derisk-v2 showed that filling padding with random token ids instead of '
        "<code>&lt;|im_end|&gt;</code> cuts step time by 12% when every micro-batch is padded to 8,192 tokens, but by "
        "about 1% when each micro-batch is padded only to its longest sequence (batchmax) on Tulu-3. This screen asks "
        "which natural SFT dataset would make the batchmax gain largest. Each candidate is sampled from the Hub, "
        "tokenized exactly as the recipe does, and run through derisk's data order; a model calibrated on the "
        "derisk-v1/v2 routing counts and step times then predicts EP imbalance and step time with "
        "<code>&lt;|im_end|&gt;</code> pads and with random pads. No GPU runs yet: the top candidates go to a short "
        "A/B once approved.</p>",
    ]

    base8 = res["tulu3"]["caps"]["8192"]["screen"]["lbs2"]
    order = sorted(
        (d for d in ds if d["key"] != "tulu3"),
        key=lambda d: -res[d["key"]]["caps"]["8192"]["screen"]["lbs2"]["saving_frac"]["mean"],
    )
    findings = []
    for d in order:
        s = res[d["key"]]["caps"]["8192"]["screen"]["lbs2"]
        findings.append(
            f"<b>{_esc(d['label'])}</b> at truncation 8,192: padding {_pct(s['pad_fraction']['mean'])}, "
            f"{s['tokens_per_microbatch']['mean']:,.0f} tokens per micro-batch, EP imbalance "
            f"{s['imbalance_eos']['mean']:.2f}× → {s['imbalance_random']['mean']:.2f}× with random pads, predicted "
            f"step time {s['step_time_eos']['mean']:.2f} s → {s['step_time_random']['mean']:.2f} s "
            f"(<b>−{_pct(s['saving_frac']['mean'])}</b>, {s['saving_s']['mean']:.2f} s)."
        )
    tulu_measured = screen["calibration"]["runs"]
    measured_saving = (
        tulu_measured["batchmax_warm/baseline"]["step_time_measured"]
        - tulu_measured["batchmax_warm/random"]["step_time_measured"]
    ) / tulu_measured["batchmax_warm/baseline"]["step_time_measured"]
    derisk = res["tulu3"]["caps"]["4096"]["screen"]["lbs2_derisk_order"]
    findings.append(
        f"<b>Tulu-3 baseline</b>: {_pct(base8['saving_frac']['mean'])} predicted at 8,192; at derisk's 4,096 "
        f"{_pct(derisk['saving_frac'])} predicted vs {_pct(measured_saving)} measured in derisk-v2 (warm pass)."
    )
    parts.append(
        '<h2 id="findings">Findings</h2><div class="findings"><ul>'
        + "".join(f"<li>{f}</li>" for f in findings)
        + "</ul></div>"
    )
    notes = _HERE / "notes.html"
    if notes.exists():
        parts.append(f'<h2 id="interpretation">Interpretation</h2><div class="panel">{notes.read_text()}</div>')

    parts.append(
        '<h2 id="ranking">Ranking</h2><div data-chart="saving"></div>'
        '<div class="grid2"><div data-chart="padshare"></div><div data-chart="tokens"></div></div>'
        '<div class="grid2"><div data-chart="imbalance"></div><div data-chart="lbs"></div></div>'
    )
    rows = []
    for cap in reversed(CAPS):
        for d in ds:
            s = res[d["key"]]["caps"][cap]["screen"]["lbs2"]
            rows.append(
                [
                    f"{_esc(d['label'])} · {int(cap):,}",
                    _pct(s["pad_fraction"]["mean"]),
                    f"{s['tokens_per_microbatch']['mean']:,.0f}",
                    f"{s['pad_tokens_per_microbatch']['mean']:,.0f}",
                    f"{s['imbalance_eos']['mean']:.2f}× → {s['imbalance_random']['mean']:.2f}×",
                    f"{s['step_time_eos']['mean']:.2f} s",
                    _range(s["saving_s"], ".2f", " s"),
                    _range(s["saving_frac"], ".1f", "%", 100),
                    _pct(s["effective_mfu_gain"]["mean"]),
                    _pct(s["saving_frac_text_like"]["mean"]),
                ]
            )
    parts.append(
        _table(
            [
                "Dataset · truncation",
                "Padding",
                "Tokens / micro-batch",
                "Pad tokens / micro-batch",
                "EP imbalance, &lt;|im_end|&gt; → random",
                "Step time, &lt;|im_end|&gt;",
                "Saving",
                "Saving %",
                "Effective-MFU gain",
                "Saving, text-like pads",
            ],
            rows,
            "Last column: upper bound if pads route like text (derisk-v2.3b's visible natural filler); see the held-out "
            "check below before trusting it. 2 sequences per GPU, 8 GPUs, global batch 128 (derisk's setup). Mean over 20 random 2,560-sample orders "
            "(5th–95th percentile below each saving; this is sampling spread, not model error). Tokens are per GPU. "
            "Effective-MFU gain = real tokens per second with random pads ÷ with <code>&lt;|im_end|&gt;</code> pads − 1.",
        )
    )

    parts.append('<h2 id="lengths">Length distributions</h2><div data-chart="cdf"></div>')
    rows = []
    for d in ds:
        r = res[d["key"]]
        lf = r["length_full"]
        rows.append(
            [
                _esc(d["label"]),
                f"{metas.get(d['key'], {}).get('rows_total', 939343):,}",
                f"{r['samples']:,}",
                f"{lf['mean']:,.0f}",
                f"{lf['p50']:,.0f}",
                f"{lf['p90']:,.0f}",
                f"{lf['p99']:,.0f}",
                f"{lf['max']:,.0f}",
                _pct(r["caps"]["4096"]["truncated_share"]),
                _pct(r["caps"]["8192"]["truncated_share"]),
                f"{r['caps']['8192']['label_tokens']['mean']:,.0f}",
                _pct(r["caps"]["8192"]["no_label_share"], 2),
            ]
        )
    parts.append(
        _table(
            [
                "Dataset",
                "Rows in split",
                "Sampled",
                "Mean",
                "Median",
                "p90",
                "p99",
                "Max",
                "Cut at 4,096",
                "Cut at 8,192",
                "Loss tokens @8,192",
                "No loss tokens @8,192",
            ],
            rows,
            "Untruncated token counts per sample as <code>ChatDataset</code> renders them. For IF-Chat the Hub "
            "parquet export holds 315,457 of the split's 426,009 rows. Every sample tokenized; none rejected.",
        )
    )
    rows = []
    for d in ds:
        for key, groups in res[d["key"]].get("breakdown", {}).items():
            if key == "language":
                continue
            for value, g in groups.items():
                rows.append(
                    [
                        f"{_esc(d['label'])} · {_esc(key)} = {_esc(value)}",
                        _pct(g["share"], 0),
                        f"{g['mean_full']:,.0f}",
                        f"{g['median_full']:,.0f}",
                        f"{g['mean_8192']:,.0f}",
                    ]
                )
    parts.append(
        '<h3 style="margin-top:18px">Length by subset</h3>'
        + _table(["Subset", "Share", "Mean", "Median", "Mean @8,192"], rows)
    )

    rows = []
    for d in ds:
        meta = metas.get(d["key"])
        if not meta:
            continue
        samples = [json.loads(line) for line in (_HERE / "samples" / f"{d['key']}.jsonl").open()]
        for key, summary in meta["population"].items():
            if "counts" not in summary or key in ("toxic",):
                continue
            total = sum(summary["counts"].values())
            for value, count in sorted(summary["counts"].items(), key=lambda kv: -kv[1])[:4]:
                share = np.mean([str(s.get(key)) == value for s in samples])
                rows.append(
                    [f"{_esc(d['label'])} · {_esc(key)} = {_esc(value)}", _pct(count / total), _pct(float(share))]
                )
        rows.append(
            [
                f"{_esc(d['label'])} · read",
                f"{len(meta['row_groups_drawn'])} of {meta['row_groups_total']} row groups",
                f"{meta['bytes_downloaded'] / 1e6:,.0f} MB",
            ]
        )
    parts.append(
        '<h2 id="sampling">Are the samples representative?</h2><p class="note">Rows come from randomly drawn parquet '
        "row groups across all files of each split (probability ∝ rows), a random subset of each. The split-wide "
        "shares come from reading the small metadata columns of every row.</p>"
        + _table(["Category", "Whole split", "Sample"], rows)
    )

    rows = []
    for d in ds:
        for cap in CAPS:
            for b in (2, 4):
                s = res[d["key"]]["caps"][cap]["screen"][f"lbs{b}"]
                rows.append(
                    [
                        f"{_esc(d['label'])} · {b} per GPU · {int(cap):,}",
                        _pct(s["pad_fraction"]["mean"]),
                        f"{s['tokens_per_microbatch']['mean']:,.0f}",
                        f"{s['max_microbatch_tokens']['mean']:,.0f}",
                        f"{s['imbalance_eos']['mean']:.2f}× → {s['imbalance_random']['mean']:.2f}×",
                        f"{s['saving_s']['mean']:.2f} s",
                        _pct(s["saving_frac"]["mean"]),
                    ]
                )
    parts.append(
        '<h2 id="lbs">Sequences per micro-batch</h2><p class="note">The model is calibrated on 8 micro-batches per step; '
        "with 4 sequences per GPU a step has 4 larger ones, so treat these as rough.</p>"
        + _table(
            [
                "Dataset · sequences per GPU · truncation",
                "Padding",
                "Tokens / micro-batch",
                "Largest micro-batch",
                "EP imbalance",
                "Saving",
                "Saving %",
            ],
            rows,
        )
    )

    cal = screen["calibration"]
    rows = [
        [
            _esc(key),
            f"{run['padded_tokens_per_step']:,.0f}",
            f"{run['imbalance_measured']:.2f}×",
            f"{run['step_time_measured']:.2f} ± {run['step_time_sem']:.2f} s",
            f"{run['step_time_fitted']:.2f} s",
        ]
        for key, run in cal["runs"].items()
    ]
    parts.append(
        '<h2 id="model">Model and calibration</h2><p class="note">Per dispatch (one micro-batch index on all 8 GPUs) '
        "with pad fraction <i>f</i>, the busiest GPU in MoE layer <i>l</i> gets "
        "<i>I<sub>l</sub></i> = 8 · max<sub>g</sub>[(1 − <i>f</i>) · real<sub>lg</sub> + <i>f</i> · pad<sub>lg</sub>] "
        "times the mean load, with real<sub>lg</sub> / pad<sub>lg</sub> the measured shares of real-token / pad-token "
        "assignments per GPU (Tulu-3 real tokens; <code>&lt;|im_end|&gt;</code> or random pads). Step time = A + B·N + "
        "K·Σ<sub>dispatch, layer</sub> (6/8)·T·(<i>I</i> − 1), with N padded tokens per step and T per dispatch. "
        f"Fit: K = {cal['k_seconds']:.3g} s per extra expert assignment on the busiest GPU (from the three "
        f"pad-fixed8192 runs), A = {cal['a_seconds']:.2f} s, B = {cal['b_seconds_per_token'] * 1e6:.2f} µs per "
        "token (with the warm pad-batchmax runs). The sampler replay reproduces derisk-v1's recorded micro-batch "
        "shapes exactly. Real-token routing of other datasets may differ from Tulu-3's, but at these pad fractions "
        "the pad hot spot dominates: tripling the real-token skew changes the imbalance removed by random pads by "
        "under 0.02.</p>"
        + _table(["derisk run", "Padded tokens / step", "EP imbalance", "Step time, measured", "Fitted"], rows)
    )
    rows = [
        [
            _esc(label),
            _esc({"random": "random", "text_like": "text-like"}[h["pad_kind"]]),
            _pct(h["saving_predicted"]),
            f"{_pct(h['saving_measured'])} ± {_pct(h['saving_measured_sem'])}",
        ]
        for label, h in cal["holdout"].items()
    ]
    parts.append(
        '<h3 style="margin-top:18px">Check against the derisk-v2 pad-batchmax variants</h3>'
        + _table(
            ["Variant (Tulu-3, truncation 4,096, warm pass)", "Modeled as", "Predicted saving", "Measured saving"],
            rows,
            "Measured: mean per-step difference to the v1 baseline over steps 20–39 ± its standard error (both runs "
            "see the same data in the same order, so steps are paired). v2.1 enters the fit; v2.3a/b do not. The "
            "model overpredicts at these 2k-token micro-batches, most for text-like pads.",
        )
    )
    parts.append(
        '<h2 id="repro">Reproduce</h2><pre>source experiments/derisk-v1/env.sh\n'
        "python experiments/choose-datasets-v1/sample_datasets.py --output-dir experiments/choose-datasets-v1/samples\n"
        "python experiments/choose-datasets-v1/make_generation_template.py\n"
        "python experiments/choose-datasets-v1/test_ep_model.py\n"
        "python experiments/choose-datasets-v1/screen.py --workers 64\n"
        "python experiments/choose-datasets-v1/make_report.py --output hub/choose-datasets-v1/index.html</pre>"
    )
    return {"charts": _charts(screen, lengths)}, "\n".join(parts)


def main() -> None:
    """Load the screen results and write the self-contained HTML report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=_HERE / "runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    screen = json.loads((args.runs_dir / "screen.json").read_text())
    lengths = {k: dict(np.load(args.runs_dir / "lengths" / f"{k}.npz")) for k in screen["datasets"]}
    metas = {
        p.name.removesuffix(".meta.json"): json.loads(p.read_text()) for p in (_HERE / "samples").glob("*.meta.json")
    }
    payload, body = build(screen, lengths, metas)
    page = (
        _TEMPLATE.read_text()
        .replace("{{TITLE}}", "choose-datasets-v1: where do random pad tokens pay off?")
        .replace("{{BODY_HTML}}", body)
        .replace("{{PAYLOAD_JSON}}", json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    logger.info("wrote %s (%.0f KiB)", args.output, len(page) / 1024)


if __name__ == "__main__":
    main()
