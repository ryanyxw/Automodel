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

"""Build the derisk-v1 HTML report from the run directories (one self-contained file, no network).

Every number in the report is computed here from the run outputs (``probe_steps.jsonl``,
``routing.npz``, ``summary.json``, ``gpu_telemetry.csv``, ``run_metadata.json``, ``data_stats.json``).
Hand-written interpretation, if any, is read from ``notes.html`` next to this script.

Usage::

    python experiments/derisk-v1/make_report.py --runs-dir experiments/derisk-v1/runs \
        --output hub/derisk-v1/index.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
ARMS = (
    {
        "key": "pad_batchmax",
        "label": "pad-batchmax",
        "desc": "each micro-batch padded to its longest sequence",
        "color": "--series-1",
    },
    {
        "key": "pad_fixed8192",
        "label": "pad-fixed8192",
        "desc": "every micro-batch padded to 8,192 tokens",
        "color": "--series-2",
    },
)
WARMUP_STEPS = 2


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _esc(value: object) -> str:
    return html.escape(str(value))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two distributions over experts."""
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    mid = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)


def _top_share(shares: np.ndarray, k: int) -> np.ndarray:
    """Fraction of mass on the ``k`` largest entries along the last axis."""
    total = np.maximum(shares.sum(axis=-1), 1e-12)
    return np.sort(shares, axis=-1)[..., ::-1][..., :k].sum(axis=-1) / total


def _load_arm(run_dir: Path) -> dict | None:
    """Load one arm's outputs, or ``None`` if the run did not finish writing them."""
    steps_path = run_dir / "probe_steps.jsonl"
    if not steps_path.exists():
        logger.warning("no probe_steps.jsonl in %s; skipping arm", run_dir)
        return None
    arm = {"dir": run_dir, "records": _read_jsonl(steps_path)}
    for name in ("summary.json", "run_metadata.json", "launch.json", "exit.json"):
        path = run_dir / name
        arm[name.split(".")[0]] = json.loads(path.read_text()) if path.exists() else {}
    arm["foreign_gpu"] = _foreign_gpu_processes(run_dir / "gpu_processes.csv")
    log_path = run_dir / "train.log"
    urls = (
        [
            line.split("View run at", 1)[1].strip()
            for line in log_path.read_text(errors="replace").splitlines()
            if "View run at" in line
        ]
        if log_path.exists()
        else []
    )
    arm["wandb_url"] = urls[0] if urls else None
    config_path = run_dir / "resolved_config.yaml"
    arm["resolved_config"] = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    arm["routing"] = _routing_view(run_dir / "routing.npz") if (run_dir / "routing.npz").exists() else None
    arm["telemetry"] = _telemetry_view(run_dir / "gpu_telemetry.csv", arm["records"])
    return arm


def _foreign_gpu_processes(path: Path) -> dict[str, float]:
    """Max GPU memory (MiB) per non-derisk process seen in ``gpu_processes.csv`` during a run."""
    if not path.exists():
        return {}
    seen: dict[str, float] = {}
    with path.open() as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) < 4 or "automodel-cu12" in row[2]:
                continue
            try:
                seen[row[2].strip()] = max(seen.get(row[2].strip(), 0.0), float(row[3]))
            except ValueError:
                continue
    return seen


def _routing_view(path: Path) -> dict:
    """Aggregate the router counts of one run into report-ready arrays."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata"]))
    counts = data["counts"].astype(np.float64)  # [steps, microbatches, layers, categories, experts]
    n_steps, n_micro, n_layers, n_cat, n_experts = counts.shape
    ep, top_k = meta["ep_size"], meta["top_k"]
    per_rank = n_experts // ep
    real = counts[:, :, :, 0, :]
    pad = counts[:, :, :, 1:, :].sum(axis=3)
    real_le, pad_le = real.sum(axis=(0, 1)), pad.sum(axis=(0, 1))
    real_share = real_le / np.maximum(real_le.sum(axis=-1, keepdims=True), 1)
    pad_share = pad_le / np.maximum(pad_le.sum(axis=-1, keepdims=True), 1)
    total_le = real_le + pad_le
    rank_share = lambda x: x.reshape(n_layers, ep, per_rank).sum(-1) / np.maximum(x.sum(-1, keepdims=True), 1)
    loads = (real + pad).reshape(n_steps, n_micro, n_layers, ep, per_rank).sum(-1)
    imbalance = loads.max(-1) / np.maximum(loads.mean(-1), 1e-9)  # [steps, microbatches, layers]
    hot_rank = loads.argmax(-1)
    bucket = counts[:, :, :, 1:, :].sum(axis=(0, 1))  # [layers, pad buckets, experts]
    bucket_tokens = bucket[0].sum(-1) / top_k
    real_top_sets = [set(np.argsort(real_share[layer])[::-1][:top_k].tolist()) for layer in range(n_layers)]
    pad_top = [np.argsort(pad_share[layer])[::-1][:top_k].tolist() for layer in range(n_layers)]
    return {
        "meta": meta,
        "layer_ids": data["layer_ids"].tolist(),
        "n_steps": n_steps,
        "n_experts": n_experts,
        "ep": ep,
        "top_k": top_k,
        "tokens_real": float(real_le[0].sum() / top_k),
        "tokens_pad": float(pad_le[0].sum() / top_k),
        "real_share": real_share,
        "pad_share": pad_share,
        "total_share": total_le / np.maximum(total_le.sum(-1, keepdims=True), 1),
        "real_top_share": _top_share(real_share, top_k),
        "pad_top_share": _top_share(pad_share, top_k),
        "pad_top_experts": pad_top,
        "pad_on_real_top": [float(pad_share[layer, list(real_top_sets[layer])].sum()) for layer in range(n_layers)],
        "js_pad_real": [_js_divergence(pad_share[layer], real_share[layer]) for layer in range(n_layers)],
        "rank_share_pad": rank_share(pad_le),
        "rank_share_real": rank_share(real_le),
        "rank_share_all": rank_share(total_le),
        "imbalance": imbalance,
        "imbalance_layer_mean": imbalance.mean(axis=(0, 1)),
        "imbalance_layer_max": imbalance.max(axis=(0, 1)),
        "imbalance_step_mean": imbalance.mean(axis=(1, 2)),
        "imbalance_step_max": imbalance.max(axis=(1, 2)),
        "hot_rank": hot_rank,
        "bucket_names": meta["category_names"][1:],
        "bucket_tokens": bucket_tokens,
        "bucket_top_share": [
            float(np.mean(_top_share(bucket[:, b, :], top_k))) if bucket_tokens[b] > 0 else None
            for b in range(bucket.shape[1])
        ],
        "bucket_js_real": [
            float(np.mean([_js_divergence(bucket[layer, b], real_le[layer]) for layer in range(n_layers)]))
            if bucket_tokens[b] > 0
            else None
            for b in range(bucket.shape[1])
        ],
        "microbatch_shapes": data["microbatch_shapes"],  # [steps, ranks, microbatches, 3]
    }


def _telemetry_view(path: Path, records: list[dict]) -> dict | None:
    """Mean GPU utilization / power / SM clock per second over the training steps."""
    if not path.exists() or not records or "timestamp" not in records[0]:
        return None
    with path.open() as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    col = {name: header.index(name) for name in header}
    step_end = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in records]
    first_start = step_end[0].timestamp() - float(records[0]["probe/step_time_wall_s"])
    last_end = step_end[-1].timestamp()
    local_tz = datetime.now().astimezone().tzinfo
    per_second: dict[int, list[tuple[float, float, float]]] = {}
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        try:
            stamp = datetime.strptime(row[col["timestamp"]].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=local_tz)
            sample = (
                float(row[col["utilization.gpu [%]"]]),
                float(row[col["power.draw [W]"]]),
                float(row[col["clocks.current.sm [MHz]"]]),
            )
        except ValueError:
            continue
        t = stamp.timestamp()
        if first_start <= t <= last_end:
            per_second.setdefault(int(t - first_start), []).append(sample)
    if not per_second:
        return None
    seconds = sorted(per_second)
    mean = lambda i: [float(np.mean([s[i] for s in per_second[t]])) for t in seconds]
    return {"t": seconds, "util": mean(0), "power": mean(1), "sm_clock": mean(2)}


def _series(records: list[dict], key: str, scale: float = 1.0) -> list[list[float | None]]:
    return [[r["step"], None if r.get(key) is None else float(r[key]) * scale] for r in records]


def _steady(records: list[dict], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in records if r["step"] >= WARMUP_STEPS and r.get(key) is not None], dtype=float)


def _mean_std(records: list[dict], key: str) -> tuple[float, float]:
    values = _steady(records, key)
    return (float(values.mean()), float(values.std())) if values.size else (float("nan"), float("nan"))


def _line(title: str, subtitle: str, arms: list[dict], key: str, y_label: str, y_fmt: str, **extra) -> dict:
    scale = extra.pop("scale", 1.0)
    return {
        "type": "line",
        "title": title,
        "subtitle": subtitle,
        "series": [
            {"label": a["label"], "color": a["color"], "points": _series(a["records"], key, scale)} for a in arms
        ],
        "x": {"label": "Step", "fmt": "int"},
        "y": {"label": y_label, "fmt": y_fmt, **extra.pop("y", {})},
        "shade": [0, WARMUP_STEPS - 1],
        "shadeLabel": "warmup",
        **extra,
    }


def _heat(title: str, subtitle: str, matrix: np.ndarray, layer_ids: list[int], experts_per_rank: int, **extra) -> dict:
    n_experts = matrix.shape[1]
    return {
        "type": "heatmap",
        "title": title,
        "subtitle": subtitle,
        "rows": [str(layer) for layer in layer_ids],
        "cols": [str(e) for e in range(n_experts)],
        "values": np.round(matrix, 6).tolist(),
        "vmax": float(extra.pop("vmax", matrix.max())),
        "fmt": "frac_pct2",
        "scale": "sqrt",
        "rowName": "Layer",
        "colName": "Expert",
        "colSuffix": [f" (EP rank {e // experts_per_rank})" for e in range(n_experts)],
        "rowTitle": "MoE layer",
        "colTitle": f"Expert (grouped by EP rank, {experts_per_rank} per GPU)",
        "groupEvery": experts_per_rank,
        "groupLabels": [f"EP{r}" for r in range(n_experts // experts_per_rank)],
        "valueLabel": "of the layer's assignments",
        **extra,
    }


def build_payload_and_body(
    arms: list[dict], data_stats: dict | None, notes_html: str, extras: dict
) -> tuple[dict, str]:
    """Compute every chart spec and the HTML body for the report.

    Args:
        arms: Loaded main arms (``_load_arm`` output merged with the ``ARMS`` entry), in ``ARMS`` order.
        data_stats: ``data_stats.json`` contents, if present.
        notes_html: Hand-written interpretation HTML, or an empty string.
        extras: Optional diagnostics: ``shape_overhead`` (bench_shape_overhead.py JSON), ``two_pass``
            (a loaded two-pass pad_batchmax run), ``oom`` (the out-of-memory line from the 4-per-GPU smoke test).

    Returns:
        The JSON payload (chart specs) and the report body HTML.
    """
    charts: dict[str, dict] = {}
    by_key = {a["key"]: a for a in arms}
    a_arm, b_arm = by_key.get("pad_batchmax"), by_key.get("pad_fixed8192")
    parts: list[str] = []

    # ---- header ------------------------------------------------------------------------------
    meta = arms[0]["run_metadata"]
    packages = meta.get("packages", {})
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = arms[0]["launch"].get("git_commit", "")[:10]
    parts.append(
        '<header class="page"><div>'
        "<h1>Does padding change MFU? Nemotron 3 Nano SFT on Tulu-3 (derisk-v1)</h1>"
        f'<div class="meta">Generated {_esc(generated)} · host {_esc(meta.get("hostname", ""))} · '
        f"{_esc(meta.get('world_size', ''))}× {_esc(meta.get('gpu_name', ''))} · git {_esc(commit)} · "
        "wandb project <code>derisk-v1</code></div></div>"
        '<button type="button" id="theme-toggle">Theme: auto</button></header>'
    )
    parts.append(
        '<p class="lede">Two otherwise identical 20-step full-parameter SFT runs of '
        "<code>nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16</code> on the same 2,560 Tulu-3 conversations, "
        "run one after the other on the same 8 GPUs. Only the padding rule differs: "
        f"<b>{_esc(ARMS[0]['label'])}</b> ({_esc(ARMS[0]['desc'])}) vs "
        f"<b>{_esc(ARMS[1]['label'])}</b> ({_esc(ARMS[1]['desc'])}).</p>"
    )
    parts.append(
        '<nav class="toc"><a href="#findings">Findings</a><a href="#setup">Setup</a><a href="#mfu">MFU</a>'
        '<a href="#throughput">Throughput</a><a href="#shapes">Per-shape costs</a><a href="#padding">Padding</a><a href="#routing">Expert routing</a>'
        '<a href="#memory">Memory</a><a href="#telemetry">GPU telemetry</a><a href="#profiling">Nsight handoff</a>'
        '<a href="#repro">Reproduce</a><a href="#appendix">Per-step table</a></nav>'
    )

    # ---- findings (computed) -----------------------------------------------------------------
    findings = []
    if a_arm and b_arm:
        ar, br = a_arm["records"], b_arm["records"]
        a_pm, _ = _mean_std(ar, "probe/mfu_padded_wall")
        b_pm, _ = _mean_std(br, "probe/mfu_padded_wall")
        a_em, _ = _mean_std(ar, "probe/mfu_effective_wall")
        b_em, _ = _mean_std(br, "probe/mfu_effective_wall")
        a_st, _ = _mean_std(ar, "probe/step_time_wall_s")
        b_st, _ = _mean_std(br, "probe/step_time_wall_s")
        a_pf, _ = _mean_std(ar, "probe/pad_fraction")
        b_pf, _ = _mean_std(br, "probe/pad_fraction")
        a_tp, _ = _mean_std(ar, "probe/real_tokens_per_s_per_gpu")
        b_tp, _ = _mean_std(br, "probe/real_tokens_per_s_per_gpu")
        a_tok, _ = _mean_std(ar, "probe/tokens_padded")
        b_tok, _ = _mean_std(br, "probe/tokens_padded")
        findings.append(
            f"Padding to 8,192 multiplied the tokens processed per step by {b_tok / a_tok:.1f}× (pad fraction "
            f"{100 * a_pf:.1f}% → {100 * b_pf:.1f}%), yet the step time went {a_st:.2f} s → {b_st:.2f} s "
            f"({b_st / a_st:.2f}×); means over steps {WARMUP_STEPS}–19."
        )
        findings.append(
            f"<b>Padded MFU</b> (AutoModel's <code>mfu</code>: FLOPs of the padded shape, pads count as work): "
            f"{a_pm:.2f}% → {b_pm:.2f}%."
        )
        findings.append(
            f"<b>Effective MFU</b> (FLOPs of real tokens only): {a_em:.2f}% → {b_em:.2f}%; real tokens/s per GPU "
            f"{a_tp:,.0f} → {b_tp:,.0f} ({b_tp / a_tp:.2f}×)."
        )
        rb = b_arm["routing"]
        if rb is not None:
            k = rb["top_k"]
            pad_share = float(np.mean(rb["pad_top_share"]))
            per_step = _steady(br, f"routing/pad_top{k}_share_mean")
            headline = (
                f"Every pad token routes to the same {k} experts per layer."
                if per_step.size and per_step.min() > 0.999
                else "Pad tokens concentrate on a few experts per layer."
            )
            findings.append(
                f"<b>{headline}</b> Within a step, the top-{k} experts take "
                f"{100 * float(per_step.mean()) if per_step.size else float('nan'):.1f}% of pad-token assignments "
                f"({ARMS[1]['label']}); over all steps {100 * pad_share:.1f}% (the set shifts slightly as the router trains). "
                f"Real tokens: {100 * float(np.mean(rb['real_top_share'])):.1f}%; uniform routing: {100 * k / rb['n_experts']:.1f}%."
            )
            ra = a_arm["routing"]
            findings.append(
                "Expert-parallel load imbalance (max ÷ mean tokens per GPU per dispatch, averaged over layers and "
                f"micro-batches): {float(ra['imbalance'].mean()):.2f} → {float(rb['imbalance'].mean()):.2f}; "
                f"worst dispatch {float(ra['imbalance'].max()):.2f} → {float(rb['imbalance'].max()):.2f} "
                f"(1.00 = balanced, {rb['ep']} = one GPU gets everything)."
            )
        la = np.asarray([r["loss"] for r in ar])
        lb = np.asarray([r["loss"] for r in br])
        n = min(len(la), len(lb))
        findings.append(
            f"Loss sanity check: identical data in the same order, mean |Δloss| = {float(np.abs(la[:n] - lb[:n]).mean()):.4f} "
            "over the steps (padding should not change the math)."
        )
    parts.append('<h2 id="findings">Findings</h2><div class="findings"><ul>')
    parts.extend(f"<li>{f}</li>" for f in findings)
    parts.append("</ul></div>")

    # ---- KPI tiles ---------------------------------------------------------------------------
    tiles = [
        ("Padded MFU (wall clock)", "probe/mfu_padded_wall", "{:.2f}%", 1.0),
        ("Effective MFU (wall clock)", "probe/mfu_effective_wall", "{:.2f}%", 1.0),
        ("Step time", "probe/step_time_wall_s", "{:.2f} s", 1.0),
        ("Real tokens / s / GPU", "probe/real_tokens_per_s_per_gpu", "{:,.0f}", 1.0),
        ("Pad fraction of processed tokens", "probe/pad_fraction", "{:.1f}%", 100.0),
        ("Peak reserved memory (max rank)", "probe/mem_reserved_peak_gib_max_rank", "{:.1f} GiB", 1.0),
    ]
    parts.append(f'<div class="kpis" aria-label="Steady-state means, steps {WARMUP_STEPS}-19">')
    for label, key, pattern, scale in tiles:
        parts.append(f'<div class="tile"><div class="label">{_esc(label)}</div>')
        for a in arms:
            mean, std = _mean_std(a["records"], key)
            parts.append(
                f'<div class="row"><i class="key" style="background: var({a["color"]})"></i>'
                f'<span class="arm">{_esc(a["label"])}</span><span class="value">{_esc(pattern.format(mean * scale))}</span>'
                f'<span class="sub">± {_esc(pattern.format(std * scale).replace("%", "").replace(" s", "").replace(" GiB", ""))}</span></div>'
            )
        parts.append("</div>")
    parts.append("</div>")
    parts.append(
        f'<p class="note">Tiles: mean ± standard deviation over steps {WARMUP_STEPS}–19 (steps 0–{WARMUP_STEPS - 1} are warmup).</p>'
    )

    if notes_html:
        parts.append(f'<h2 id="interpretation">Interpretation</h2><div class="panel">{notes_html}</div>')

    # ---- setup -------------------------------------------------------------------------------
    sched = arms[0]["resolved_config"].get("step_scheduler", {})
    world = int(meta.get("world_size", 8))
    local_batch, global_batch = int(sched.get("local_batch_size", 0)), int(sched.get("global_batch_size", 0))
    accum = global_batch // max(local_batch * world, 1)
    batching = (
        f"global batch {global_batch} = {local_batch} sequences per GPU × {world} GPUs × {accum} accumulation steps; "
        f"{sched.get('max_steps')} steps = each sample once."
    )
    if extras.get("oom") and local_batch < 4:
        batching += f" (4 sequences per GPU ran out of memory in {ARMS[1]['label']}, see Memory, so both arms use {local_batch}.)"
    foreign = {name: mem for a in arms for name, mem in a.get("foreign_gpu", {}).items()}
    order_text = "Arms run one after the other on the same node; run_arm.sh refuses to start if another process is using the GPUs. "
    if foreign:
        listed = ", ".join(f"{_esc(name)} ({mem:.0f} MiB)" for name, mem in sorted(foreign.items()))
        order_text += (
            f"Present but idle during the runs (shared machine, logged every 5 s in gpu_processes.csv): {listed}."
        )
    else:
        order_text += "No other GPU processes were seen during the runs."
    cfg_rows = [
        (
            "Model",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 — 31.6B params, 52 layers (23 Mamba-2, 23 MoE, 6 attention), 128 routed experts, top-6, 1 shared expert",
        ),
        (
            "Training",
            "Full-parameter SFT, TE FusedAdam (bf16 params, fp32 master weights), lr 1e-5 cosine, FusedLinearCrossEntropy, answer-only loss",
        ),
        (
            "Parallelism",
            "FSDP2 over 8 GPUs + expert parallelism EP=8 (16 experts per GPU), activation checkpointing, DeepEP dispatch, TE attention/linear, torch grouped-GEMM experts",
        ),
        (
            "Data",
            "allenai/tulu-3-sft-mixture, random 2,560-conversation subset (shuffle seed 42), Nemotron chat template, truncated at 4,096 tokens, fixed order",
        ),
        ("Batching", _esc(batching)),
        (
            "Padding token",
            "<code>&lt;|im_end|&gt;</code> (id 11, the EOS token; the tokenizer has no pad token), label −100, attention mask 0",
        ),
        (
            "Arms",
            f"{_esc(ARMS[0]['label'])}: <code>default_collater</code> with no rounding. {_esc(ARMS[1]['label'])}: <code>pad_seq_len_divisible: 8192</code>. Nothing else differs (run_all.sh diff-checks the configs).",
        ),
        ("Order", order_text),
        ("Software", _esc(", ".join(f"{k} {v}" for k, v in packages.items() if v))),
        ("CUDA", _esc(f"torch CUDA {meta.get('torch_cuda')}, cuDNN {meta.get('cudnn')}, NCCL {meta.get('nccl')}")),
    ]
    parts.append('<h2 id="setup">Setup</h2><div class="panel"><table class="kv"><tbody>')
    parts.extend(f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>" for k, v in cfg_rows)
    parts.append("</tbody></table></div>")
    parts.append(
        '<div class="panel"><h3>How MFU is computed</h3><ul>'
        "<li>Both use AutoModel's NemotronH FLOPs formula (<code>nemotronh_flops</code>: 6× per-token GEMM FLOPs "
        "for Mamba, attention, routed top-6 + shared experts, router, and LM head, plus the causal attention term) "
        "and the H100 SXM dense BF16 peak of 989 TFLOP/s per GPU. Activation-checkpoint recompute is not counted as useful work.</li>"
        "<li><b>Padded MFU</b> applies the formula to each micro-batch's padded [batch, sequence] shape. This is exactly "
        "AutoModel's built-in <code>mfu</code> metric (the report checks they match). It counts pad tokens as work.</li>"
        "<li><b>Effective MFU</b> applies the formula to each sequence's real length and sums, so only real tokens count.</li>"
        "<li><i>Wall clock</i> divides by the step-to-step wall time (AutoModel's definition); <i>compute</i> divides by the "
        "GPU-synchronized time of the optimizer step (slowest rank), excluding data loading. Both are in the table views.</li>"
        "<li>What the padded formula gets wrong for this model: pad tokens are masked out of attention by TE "
        "(<code>padding_causal</code>), so their attention FLOPs are not executed; but they are fully processed by Mamba, "
        "the shared expert, the router, and the routed experts (Nemotron V3 does not pass its padding mask to the MoE), "
        "so most of the padded FLOPs are really executed.</li></ul></div>"
    )
    parts.append(
        '<div class="panel"><h3>Environment deviation (read this)</h3><p class="note">The host runs NVIDIA driver 550 '
        "(CUDA 12.4). The lockfile's torch 2.10.0+cu130 and the NGC nemo-automodel 26.08 container (CUDA 13.3) cannot "
        "run on it, so the runs use a uv venv with the lockfile versions except PyPI's torch 2.10.0 built for CUDA 12.8, "
        f"cuDNN {_esc(packages.get('nvidia-cudnn-cu12'))}, TE 2.19 / causal-conv1d / mamba-ssm compiled for sm_90 with nvcc 12.4, and DeepEP at the lockfile "
        "revision built intranode-only (no NVSHMEM, no HybridEP). See <code>experiments/derisk-v1/setup_env.sh</code>. "
        "Absolute MFU may differ from the container stack; the A/B comparison uses the same stack for both arms.</p></div>"
    )

    # ---- MFU ---------------------------------------------------------------------------------
    parts.append('<h2 id="mfu">MFU throughout training</h2>')
    charts["mfu_padded"] = _line(
        "Padded MFU per step",
        "Pads counted as work (AutoModel's mfu), wall-clock step time",
        arms,
        "probe/mfu_padded_wall",
        "MFU (%)",
        "pct2",
        y={"min": 0, "tickFmt": "pct1"},
    )
    charts["mfu_effective"] = _line(
        "Effective MFU per step",
        "Only real tokens counted, wall-clock step time",
        arms,
        "probe/mfu_effective_wall",
        "MFU (%)",
        "pct2",
        y={"min": 0, "tickFmt": "pct1"},
    )
    charts["mfu_padded_compute"] = _line(
        "Padded MFU, GPU-synchronized step time",
        "Excludes data loading and inter-step host work",
        arms,
        "probe/mfu_padded_compute",
        "MFU (%)",
        "pct2",
        y={"min": 0, "tickFmt": "pct1"},
    )
    charts["mfu_effective_compute"] = _line(
        "Effective MFU, GPU-synchronized step time",
        "Excludes data loading and inter-step host work",
        arms,
        "probe/mfu_effective_compute",
        "MFU (%)",
        "pct2",
        y={"min": 0, "tickFmt": "pct1"},
    )
    parts.append(
        '<div class="grid2"><div data-chart="mfu_padded"></div><div data-chart="mfu_effective"></div>'
        '<div data-chart="mfu_padded_compute"></div><div data-chart="mfu_effective_compute"></div></div>'
    )
    if a_arm and b_arm:
        diffs = [
            abs(float(r["mfu"]) - float(r["probe/mfu_padded_wall"]))
            for a in arms
            for r in a["records"]
            if r.get("mfu") is not None
        ]
        parts.append(
            f"<p class=\"note\">Consistency check: AutoModel's <code>mfu</code> and the probe's padded MFU differ by at most "
            f"{max(diffs) if diffs else float('nan'):.2e} percentage points across all steps.</p>"
        )

    # ---- throughput --------------------------------------------------------------------------
    parts.append('<h2 id="throughput">Step time and throughput</h2>')
    charts["step_time"] = _line(
        "Step time",
        "Wall clock between optimizer steps",
        arms,
        "probe/step_time_wall_s",
        "Seconds",
        "s2",
        y={"min": 0, "tickFmt": "s0"},
    )
    charts["real_tps"] = _line(
        "Real tokens / s / GPU",
        "Non-padding tokens processed per second per GPU",
        arms,
        "probe/real_tokens_per_s_per_gpu",
        "Tokens / s / GPU",
        "num0",
        y={"min": 0},
    )
    charts["padded_tps"] = _line(
        "Padded tokens / s / GPU",
        "All tokens, pads included",
        arms,
        "probe/padded_tokens_per_s_per_gpu",
        "Tokens / s / GPU",
        "num0",
        y={"min": 0},
    )
    charts["tflops"] = _line(
        "Effective TFLOP/s per GPU",
        "Real-token FLOPs per second per GPU (wall clock)",
        arms,
        "probe/tflops_per_gpu_effective_wall",
        "TFLOP/s",
        "num1",
        y={"min": 0, "tickFmt": "num0"},
    )
    parts.append(
        '<div class="grid2"><div data-chart="step_time"></div><div data-chart="real_tps"></div>'
        '<div data-chart="padded_tps"></div><div data-chart="tflops"></div></div>'
    )

    # ---- dynamic shapes ----------------------------------------------------------------------
    shape = extras.get("shape_overhead")
    two_pass = extras.get("two_pass")
    if shape or two_pass:
        parts.append('<h2 id="shapes">Why pad-batchmax is slow early on: per-shape costs</h2>')
    if shape:
        res = shape["results"]
        attn, cce = res["te_attention"], res["cut_cross_entropy"]
        parts.append(
            '<div class="panel"><p>Padding each micro-batch to its own longest sequence gives almost every micro-batch a '
            "sequence length the process has not seen yet. Two kernels on this model build or tune something per shape. "
            f"Measured on one H100 (<code>bench_shape_overhead.py</code>, forward + backward at batch {shape['batch']}, "
            f"{len(shape['lengths'])} distinct lengths from {min(shape['lengths'])} to {max(shape['lengths'])}):</p>"
            '<table><thead><tr><th class="text">Kernel</th><th>First call at a new length</th><th>Repeat call</th>'
            "<th>Cost per new length</th></tr></thead><tbody>"
            f'<tr><td class="text">TE / cuDNN fused attention (<code>padding_causal</code>)</td><td>{attn["cold_mean_ms"]:,.1f} ms</td>'
            f"<td>{attn['warm_mean_ms']:,.1f} ms</td><td>{attn['overhead_per_new_shape_ms']:,.1f} ms</td></tr>"
            f'<tr><td class="text">cut-cross-entropy (Triton autotune per token-count bin)</td><td>{cce["cold_mean_ms"]:,.1f} ms</td>'
            f"<td>{cce['warm_mean_ms']:,.1f} ms</td><td>{cce['overhead_per_new_shape_ms']:,.1f} ms</td></tr>"
            '</tbody></table><p class="note">cuDNN builds an attention execution plan per sequence length (shared by the six '
            "attention layers and reused by activation-checkpoint recompute), so the cost is paid about once per micro-batch "
            "with a new length. pad-fixed8192 has one shape and pays it once.</p></div>"
        )
    if two_pass:
        tp = two_pass["records"]
        n_first = sum(1 for r in tp if r["step"] < 20)
        first = [r for r in tp if WARMUP_STEPS <= r["step"] < 20]
        second = [r for r in tp if r["step"] >= 20]
        mean = lambda rows, key: float(np.mean([r[key] for r in rows])) if rows else float("nan")
        rows_html = []
        metrics = (
            ("Step time", "probe/step_time_wall_s", "{:.2f} s"),
            ("Padded MFU", "probe/mfu_padded_wall", "{:.2f}%"),
            ("Effective MFU", "probe/mfu_effective_wall", "{:.2f}%"),
            ("Real tokens / s / GPU", "probe/real_tokens_per_s_per_gpu", "{:,.0f}"),
        )
        b_ref = b_arm["records"] if b_arm else []
        for label, key, pattern in metrics:
            rows_html.append(
                f'<tr><td class="text">{_esc(label)}</td><td>{pattern.format(mean(first, key))}</td>'
                f"<td>{pattern.format(mean(second, key))}</td><td>{pattern.format(_mean_std(b_ref, key)[0]) if b_ref else '–'}</td></tr>"
            )
        parts.append(
            '<div class="panel"><p>Diagnostic run (not part of the 20-step protocol): pad-batchmax for two passes over the '
            f"same 2,560 samples in the same order ({len(tp)} steps). Pass 2 repeats every micro-batch shape of pass 1, so the "
            "per-shape caches are warm and it shows the steady-state cost of dynamic padding.</p>"
            '<table><thead><tr><th class="text">Metric</th>'
            f"<th>pad-batchmax pass 1 (steps {WARMUP_STEPS}–{n_first - 1}, cold)</th><th>pad-batchmax pass 2 (steps 20–{len(tp) - 1}, warm)</th>"
            f"<th>{_esc(ARMS[1]['label'])} (steps {WARMUP_STEPS}–19)</th></tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table></div>"
        )
        series = [
            {
                "label": "pad-batchmax, two passes",
                "color": "--series-1",
                "points": _series(tp, "probe/step_time_wall_s"),
            }
        ]
        if b_arm:
            series.append(
                {"label": ARMS[1]["label"], "color": "--series-2", "points": _series(b_ref, "probe/step_time_wall_s")}
            )
        charts["two_pass_step_time"] = {
            "type": "line",
            "title": "Step time over two passes of the same data",
            "endLabels": False,
            "subtitle": "Shaded: pass 1 (new shapes); from step 20 every micro-batch shape repeats",
            "series": series,
            "x": {"label": "Step", "fmt": "int"},
            "y": {"label": "Seconds", "fmt": "s2", "min": 0, "tickFmt": "s0"},
            "shade": [0, 19],
            "shadeLabel": "pass 1",
        }
        series = [
            {
                "label": "pad-batchmax, two passes",
                "color": "--series-1",
                "points": _series(tp, "probe/mfu_effective_wall"),
            }
        ]
        if b_arm:
            series.append(
                {"label": ARMS[1]["label"], "color": "--series-2", "points": _series(b_ref, "probe/mfu_effective_wall")}
            )
        charts["two_pass_mfu"] = {
            "type": "line",
            "title": "Effective MFU over two passes of the same data",
            "endLabels": False,
            "subtitle": "Only real tokens counted; wall clock",
            "series": series,
            "x": {"label": "Step", "fmt": "int"},
            "y": {"label": "MFU (%)", "fmt": "pct2", "min": 0, "tickFmt": "pct1"},
            "shade": [0, 19],
            "shadeLabel": "pass 1",
        }
        parts.append(
            '<div class="grid2"><div data-chart="two_pass_step_time"></div><div data-chart="two_pass_mfu"></div></div>'
        )

    # ---- padding -----------------------------------------------------------------------------
    parts.append('<h2 id="padding">Padding</h2>')
    charts["pad_fraction"] = _line(
        "Pad fraction per step",
        "Share of processed tokens that are padding",
        arms,
        "probe/pad_fraction",
        "Pad fraction",
        "frac_pct1",
        y={"min": 0, "max": 1, "tickFmt": "frac_pct0"},
    )
    charts["loss"] = _line(
        "Training loss",
        "Sanity check: same data, same order; the curves should overlap",
        arms,
        "loss",
        "Loss",
        "num3",
        y={"tickFmt": "num2"},
    )
    parts.append('<div class="grid2"><div data-chart="pad_fraction"></div><div data-chart="loss"></div></div>')
    if data_stats:
        hist = data_stats.get("sequence_length_histogram", {})
        charts["length_hist"] = {
            "type": "bars",
            "title": "Sequence lengths in the 2,560-sample subset",
            "subtitle": f"Nemotron chat template, truncated at 4,096 tokens; mean {data_stats['sequence_length']['mean']:.0f}, median {data_stats['sequence_length']['p50']:.0f}",
            "categories": list(hist.keys()),
            "series": [{"label": "Samples", "color": "--series-1", "values": list(hist.values())}],
            "x": {"label": "Tokens"},
            "y": {"label": "Samples", "fmt": "int"},
        }
        parts.append('<div class="grid2"><div data-chart="length_hist"></div><div>')
        pa = data_stats["arms"]["pad_batchmax"]
        pb = data_stats["arms"]["pad_fixed8192"]
        parts.append(
            '<div class="panel"><h3>Predicted before training (data_stats.py)</h3><table class="kv"><tbody>'
            f"<tr><td>Real tokens (all 20 steps)</td><td>{pa['total_real_tokens']:,}</td></tr>"
            f"<tr><td>{_esc(ARMS[0]['label'])} processed tokens</td><td>{pa['total_padded_tokens']:,} ({100 * pa['pad_fraction']:.1f}% padding)</td></tr>"
            f"<tr><td>{_esc(ARMS[1]['label'])} processed tokens</td><td>{pb['total_padded_tokens']:,} ({100 * pb['pad_fraction']:.1f}% padding)</td></tr>"
            f"<tr><td>Samples at the 4,096 cap</td><td>{data_stats['samples_at_truncation_limit']}</td></tr>"
            f"<tr><td>Samples with no supervised tokens</td><td>{data_stats['samples_without_labels']} (truncation cut the answer)</td></tr>"
            "</tbody></table></div></div></div>"
        )

    # ---- routing -----------------------------------------------------------------------------
    parts.append('<h2 id="routing">Expert routing: where do pad tokens go?</h2>')
    parts.append(
        '<p class="note">Router hooks on all 23 MoE layers record every top-6 assignment, split into real tokens and '
        "pad tokens (bucketed by distance from the last real token), summed over all ranks, micro-batches and steps. "
        "Activation-checkpoint recomputation is filtered out. Nemotron V3 does not pass its padding mask to the MoE, "
        "so these are also the experts that pad tokens are dispatched to and computed on.</p>"
    )
    for a in arms:
        r = a["routing"]
        if r is None:
            continue
        layer_ids, per_rank, k = r["layer_ids"], r["n_experts"] // r["ep"], r["top_k"]
        key = a["key"]
        charts[f"topshare_{key}"] = {
            "type": "line",
            "title": f"{a['label']}: share of assignments on each layer's top-{k} experts",
            "subtitle": f"Pad tokens (accent) vs real tokens (gray); uniform routing would be {100 * k / r['n_experts']:.1f}%",
            "series": [
                {
                    "label": "Pad tokens",
                    "color": "--accent",
                    "points": [[lid, float(v)] for lid, v in zip(layer_ids, r["pad_top_share"])],
                },
                {
                    "label": "Real tokens",
                    "color": "--deemph",
                    "points": [[lid, float(v)] for lid, v in zip(layer_ids, r["real_top_share"])],
                },
            ],
            "x": {"label": "Layer", "fmt": "int", "ticks": layer_ids[::2]},
            "y": {"label": f"Top-{k} share", "fmt": "frac_pct1", "min": 0, "max": 1, "tickFmt": "frac_pct0"},
            "markers": True,
        }
        vmax = float(max(r["real_share"].max(), r["pad_share"].max()))
        charts[f"heat_real_{key}"] = _heat(
            f"{a['label']}: real-token routing",
            "Share of each layer's real-token assignments per expert",
            r["real_share"],
            layer_ids,
            per_rank,
            vmax=vmax,
        )
        charts[f"heat_pad_{key}"] = _heat(
            f"{a['label']}: pad-token routing",
            "Share of each layer's pad-token assignments per expert (same color scale)",
            r["pad_share"],
            layer_ids,
            per_rank,
            vmax=vmax,
        )
        rank_matrix = r["rank_share_all"]
        charts[f"heat_rank_{key}"] = {
            "type": "heatmap",
            "title": f"{a['label']}: share of each layer's expert work per GPU (all tokens)",
            "subtitle": f"Balanced = {100 / r['ep']:.1f}% per GPU",
            "rows": [str(lid) for lid in layer_ids],
            "cols": [str(g) for g in range(r["ep"])],
            "values": np.round(rank_matrix, 6).tolist(),
            "vmax": float(rank_matrix.max()),
            "fmt": "frac_pct1",
            "scale": "linear",
            "rowName": "Layer",
            "colName": "EP rank",
            "rowTitle": "MoE layer",
            "colTitle": "EP rank (GPU)",
            "valueLabel": "of the layer's assignments",
            "cellHeight": 12,
        }
        parts.append(f"<h3 style='margin-top:22px'>{_esc(a['label'])}</h3>")
        parts.append(f'<div data-chart="topshare_{key}"></div>')
        parts.append(
            f'<div class="grid2"><div data-chart="heat_real_{key}"></div><div data-chart="heat_pad_{key}"></div></div>'
        )
        parts.append(f'<div data-chart="heat_rank_{key}"></div>')
        rows_html = []
        for i, lid in enumerate(layer_ids):
            experts = r["pad_top_experts"][i]
            shares = ", ".join(f"{e} ({100 * r['pad_share'][i, e]:.1f}%, EP{e // per_rank})" for e in experts)
            rows_html.append(
                f'<tr><td>{lid}</td><td class="text">{_esc(shares)}</td><td>{100 * r["pad_on_real_top"][i]:.1f}%</td>'
                f"<td>{r['js_pad_real'][i]:.3f}</td><td>{r['imbalance_layer_mean'][i]:.2f}</td><td>{r['imbalance_layer_max'][i]:.2f}</td></tr>"
            )
        parts.append(
            f'<figure class="card"><div class="card-head"><div><h3>{_esc(a["label"])}: which experts pad tokens use, per layer</h3>'
            f'<p class="subtitle">Top-{k} experts by pad share (expert id, share of pad assignments, EP rank); '
            f"pad mass on the real-token top-{k}; Jensen–Shannon divergence pad vs real (0 = identical, 1 = disjoint); "
            'EP imbalance (max ÷ mean per dispatch)</p></div></div><div class="table-wrap"><table><thead><tr>'
            f'<th>Layer</th><th class="text">Pad top-{k} experts</th><th>Pad on real top-{k}</th><th>JS(pad, real)</th><th>EP imbalance mean</th><th>EP imbalance max</th>'
            "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div></figure>"
        )

    if a_arm and b_arm and a_arm["routing"] and b_arm["routing"]:
        ra, rb = a_arm["routing"], b_arm["routing"]
        layer_ids = rb["layer_ids"]
        charts["imbalance_layers"] = {
            "type": "line",
            "title": "Expert-parallel load imbalance per layer",
            "subtitle": "Max ÷ mean assignments per GPU in a dispatch, averaged over micro-batches and steps (1.0 = balanced)",
            "series": [
                {
                    "label": a["label"],
                    "color": a["color"],
                    "points": [[lid, float(v)] for lid, v in zip(layer_ids, a["routing"]["imbalance_layer_mean"])],
                }
                for a in (a_arm, b_arm)
            ],
            "x": {"label": "Layer", "fmt": "int", "ticks": layer_ids[::2]},
            "y": {"label": "Imbalance", "fmt": "x2", "min": 1, "tickFmt": "x1"},
            "markers": True,
        }
        charts["imbalance_steps"] = {
            "type": "line",
            "title": "Worst EP imbalance per step",
            "subtitle": "Max over layers and micro-batches of max ÷ mean assignments per GPU",
            "series": [
                {
                    "label": a["label"],
                    "color": a["color"],
                    "points": [[s, float(v)] for s, v in enumerate(a["routing"]["imbalance_step_max"])],
                }
                for a in (a_arm, b_arm)
            ],
            "x": {"label": "Step", "fmt": "int"},
            "y": {"label": "Imbalance", "fmt": "x2", "min": 1, "tickFmt": "x1"},
            "shade": [0, WARMUP_STEPS - 1],
        }
        same_pad = [
            float(len(set(ra["pad_top_experts"][i]) & set(rb["pad_top_experts"][i])) / rb["top_k"])
            for i in range(len(layer_ids))
        ]
        parts.append("<h3 style='margin-top:22px'>Both arms</h3>")
        parts.append(
            '<div class="grid2"><div data-chart="imbalance_layers"></div><div data-chart="imbalance_steps"></div></div>'
        )
        bucket_rows = []
        for name, tokens_a, tokens_b, share_b, js_b in zip(
            rb["bucket_names"], ra["bucket_tokens"], rb["bucket_tokens"], rb["bucket_top_share"], rb["bucket_js_real"]
        ):
            bucket_rows.append(
                f"<tr><td>{_esc(name)}</td><td>{tokens_a:,.0f}</td><td>{tokens_b:,.0f}</td>"
                f"<td>{'–' if share_b is None else f'{100 * share_b:.1f}%'}</td><td>{'–' if js_b is None else f'{js_b:.3f}'}</td></tr>"
            )
        parts.append(
            '<figure class="card"><div class="card-head"><div><h3>Pad tokens by distance from the end of the text</h3>'
            f'<p class="subtitle">Distance 0 is the first pad after the last real token. Top-{rb["top_k"]} share and JS divergence '
            f"vs real tokens are for {_esc(ARMS[1]['label'])}, averaged over layers. Same top-{rb['top_k']} pad experts in both arms: "
            f'{100 * float(np.mean(same_pad)):.0f}% of layer-slots.</p></div></div><div class="table-wrap"><table><thead><tr>'
            f"<th>Pad distance</th><th>{_esc(ARMS[0]['label'])} pad tokens</th><th>{_esc(ARMS[1]['label'])} pad tokens</th>"
            f"<th>Top-{rb['top_k']} share</th><th>JS(bucket, real)</th></tr></thead><tbody>"
            + "".join(bucket_rows)
            + "</tbody></table></div></figure>"
        )

    # ---- memory ------------------------------------------------------------------------------
    parts.append('<h2 id="memory">Memory</h2>')
    mem = {}
    for a in arms:
        per_rank = np.asarray(
            [r["per_rank_mem_reserved_peak_gib"] for r in a["records"] if "per_rank_mem_reserved_peak_gib" in r]
        )
        mem[a["key"]] = per_rank.max(axis=0) if per_rank.size else None
    if all(v is not None for v in mem.values()):
        n_ranks = len(next(iter(mem.values())))
        charts["mem_ranks"] = {
            "type": "bars",
            "title": "Peak reserved GPU memory per rank",
            "subtitle": "Max over all steps; ranks hosting the pad-token experts carry more expert activations",
            "categories": [str(i) for i in range(n_ranks)],
            "series": [
                {"label": a["label"], "color": a["color"], "values": [float(v) for v in mem[a["key"]]]} for a in arms
            ],
            "x": {"label": "Rank"},
            "y": {"label": "GiB", "fmt": "gib1", "tickFmt": "num0"},
        }
        parts.append('<div data-chart="mem_ranks"></div>')
    oom = extras.get("oom")
    if oom:
        parts.append(
            f'<p class="note">With 4 sequences per GPU, {_esc(ARMS[1]["label"])} ran out of memory in the routed experts\' grouped '
            f"GEMM on one GPU only, the one hosting that layer's pad-token experts: <code>{_esc(oom)}</code></p>"
        )

    # ---- telemetry ---------------------------------------------------------------------------
    parts.append('<h2 id="telemetry">GPU telemetry during training</h2>')
    tele = [a for a in arms if a["telemetry"]]
    if tele:
        for metric, title, label, fmt_name in (
            ("util", "GPU utilization (mean of 8 GPUs)", "Utilization (%)", "num0"),
            ("power", "Power draw (mean of 8 GPUs)", "Watts", "num0"),
        ):
            charts[f"tele_{metric}"] = {
                "type": "line",
                "title": title,
                "subtitle": "nvidia-smi every 500 ms, from the start of step 0 to the end of the last step",
                "series": [
                    {
                        "label": a["label"],
                        "color": a["color"],
                        "points": [[t, v] for t, v in zip(a["telemetry"]["t"], a["telemetry"][metric])],
                    }
                    for a in tele
                ],
                "x": {"label": "Seconds since step 0 started", "fmt": "int"},
                "y": {"label": label, "fmt": fmt_name, "min": 0},
                "endLabels": False,
            }
        parts.append('<div class="grid2"><div data-chart="tele_util"></div><div data-chart="tele_power"></div></div>')
        parts.append(
            '<p class="note">nvidia-smi utilization is the fraction of time any kernel runs, not SM occupancy.</p>'
        )

    # ---- nsight handoff ----------------------------------------------------------------------
    parts.append('<h2 id="profiling">Nsight Systems handoff</h2>')
    handoff = [
        "<li>Capture helper: <code>bash experiments/derisk-v1/profile_nsys.sh &lt;arm&gt; [start_step] [end_step] [ranks]</code> "
        "(default steps 8–9, all ranks, one <code>.nsys-rep</code> per rank under <code>runs/nsys_&lt;arm&gt;/nsys/</code>). "
        "It uses the same config as the MFU runs plus AutoModel's module-level NVTX ranges (<code>--nvtx true</code>).</li>",
        "<li>NVTX ranges always present: <code>train_step&lt;N&gt;</code> around each optimizer step and "
        "<code>microbatch&lt;i&gt;_B&lt;batch&gt;xS&lt;seq&gt;</code> around each micro-batch forward/backward.</li>",
    ]
    for a in arms:
        st = _steady(a["records"], "probe/step_time_wall_s")
        if st.size:
            handoff.append(
                f"<li>{_esc(a['label'])}: steady step time {st.mean():.2f} s (min {st.min():.2f}, max {st.max():.2f}); "
                f"a 2-step capture spans about {2 * st.mean():.0f} s of GPU time per rank.</li>"
            )
        r = a["routing"]
        if r is not None:
            hot = np.bincount(r["hot_rank"].reshape(-1), minlength=r["ep"]) / r["hot_rank"].size
            order = np.argsort(hot)[::-1]
            handoff.append(
                f"<li>{_esc(a['label'])}: rank hottest in a dispatch (share of layer-dispatches): "
                + ", ".join(f"EP{g} {100 * hot[g]:.0f}%" for g in order[:4])
                + ". Profile the hottest and the coolest rank to see the DeepEP combine wait on the straggler.</li>"
            )
    handoff.append(
        "<li>Per-rank, per-micro-batch shapes (batch, padded length, real tokens) are in <code>routing.npz → microbatch_shapes</code>; "
        "per-step, per-rank compute time and peak memory in <code>probe_steps.jsonl</code>; versions, env vars and "
        "<code>nvidia-smi topo -m</code> in <code>run_metadata.json</code>; the resolved config in <code>resolved_config.yaml</code>.</li>"
    )
    trace_dir = _HERE / "runs" / "nsys_pad_fixed8192" / "nsys"
    kernel_csvs = sorted(trace_dir.glob("*_cuda_gpu_kern_sum.csv")) if trace_dir.exists() else []
    if kernel_csvs:
        rows_html = []
        for path in kernel_csvs:
            with path.open() as f:
                kernels = list(csv.DictReader(f))
            total = sum(float(k["Total Time (ns)"]) for k in kernels)
            rank = path.name.split("_rank")[1].split("_")[0]
            for k in kernels[:4]:
                rows_html.append(
                    f'<tr><td>{_esc(rank)}</td><td class="text">{_esc(k["Name"][:90])}</td>'
                    f"<td>{float(k['Time (%)']):.1f}%</td><td>{float(k['Total Time (ns)']) / 1e9:.2f} s</td></tr>"
                )
            rows_html.append(
                f'<tr><td>{_esc(rank)}</td><td class="text"><i>all kernels</i></td><td>100%</td><td>{total / 1e9:.2f} s</td></tr>'
            )
        handoff.append(
            "<li>Sample traces (validated with the host nsys 2023.4): <code>runs/nsys_pad_fixed8192/nsys/pad_fixed8192_rank{0,1}.nsys-rep</code>, "
            'steps 2–3 of pad-fixed8192. Top GPU kernels: <div class="table-wrap"><table><thead><tr><th>Rank</th>'
            '<th class="text">Kernel</th><th>Share of kernel time</th><th>Total (2 steps)</th></tr></thead><tbody>'
            + "".join(rows_html)
            + '</tbody></table></div><span class="note">DeepEP\'s <code>cached_notify_combine</code> is a cross-rank '
            "synchronization kernel: its time is mostly waiting for the slowest GPU in each MoE dispatch.</span></li>"
        )
    handoff.append(
        "<li>Kernels to look for: DeepEP <code>intranode::dispatch/combine</code>, <code>torch._grouped_mm</code> (routed experts), "
        "mamba-ssm Triton <code>_chunk_scan*/_chunk_state*/_state_passing*</code> and causal-conv1d, TE/cuDNN fused attention, "
        "cut-cross-entropy Triton kernels, FSDP all-gather / reduce-scatter.</li>"
    )
    parts.append('<div class="panel"><ul>' + "".join(handoff) + "</ul></div>")

    # ---- reproduce ---------------------------------------------------------------------------
    parts.append('<h2 id="repro">Reproduce</h2>')
    parts.append(
        "<pre>bash experiments/derisk-v1/setup_env.sh          # uv venv (CUDA 12.8 torch) + sm_90 extension builds\n"
        "bash experiments/derisk-v1/run_all.sh            # config guard, data stats, smoke tests, both arms, report\n"
        "# or one arm:  bash experiments/derisk-v1/run_arm.sh pad_fixed8192 pad_fixed8192\n"
        "# report only: python experiments/derisk-v1/make_report.py --runs-dir experiments/derisk-v1/runs --output hub/derisk-v1/index.html</pre>"
    )
    wandb_links = [
        f'{_esc(a["label"])}: <a href="{_esc(a["wandb_url"])}">{_esc(a["wandb_url"])}</a>'
        for a in arms
        if a.get("wandb_url")
    ]
    if extras.get("two_pass") and extras["two_pass"].get("wandb_url"):
        url = extras["two_pass"]["wandb_url"]
        wandb_links.append(f'pad-batchmax two-pass diagnostic: <a href="{_esc(url)}">{_esc(url)}</a>')
    if wandb_links:
        parts.append('<p class="note">wandb runs (project derisk-v1): ' + "; ".join(wandb_links) + ".</p>")

    # ---- appendix ----------------------------------------------------------------------------
    parts.append('<h2 id="appendix">Per-step table</h2>')
    cols = [
        ("step", "Step", "{:d}"),
        ("loss", "Loss", "{:.4f}"),
        ("probe/step_time_wall_s", "Step s", "{:.2f}"),
        ("probe/step_time_compute_s", "Compute s", "{:.2f}"),
        ("probe/mfu_padded_wall", "Padded MFU %", "{:.2f}"),
        ("probe/mfu_effective_wall", "Effective MFU %", "{:.2f}"),
        ("probe/mfu_padded_compute", "Padded MFU % (sync)", "{:.2f}"),
        ("probe/mfu_effective_compute", "Effective MFU % (sync)", "{:.2f}"),
        ("probe/tokens_padded", "Padded tokens", "{:,.0f}"),
        ("probe/tokens_real", "Real tokens", "{:,.0f}"),
        ("probe/pad_fraction", "Pad fraction", "{:.3f}"),
        ("routing/ep_imbalance_all_max", "EP imbalance max", "{:.2f}"),
        ("probe/mem_reserved_peak_gib_max_rank", "Peak mem GiB", "{:.1f}"),
    ]
    for a in arms:
        body_rows = []
        for r in a["records"]:
            cells = []
            for key, _, pattern in cols:
                value = r.get(key)
                cells.append("–" if value is None else _esc(pattern.format(value)))
            body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        parts.append(
            f'<figure class="card"><h3>{_esc(a["label"])}</h3><div class="table-wrap"><table><thead><tr>'
            + "".join(f"<th>{_esc(label)}</th>" for _, label, _ in cols)
            + "</tr></thead><tbody>"
            + "".join(body_rows)
            + "</tbody></table></div></figure>"
        )

    payload = {"charts": charts}
    return payload, "\n".join(parts)


def main() -> None:
    """Load both arms, build the payload and body, and write the self-contained HTML report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=_HERE / "runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-dir", action="append", default=[], help="override as <arm_key>=<run_dir>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    overrides = dict(item.split("=", 1) for item in args.arm_dir)
    arms = []
    for spec in ARMS:
        run_dir = Path(overrides.get(spec["key"], args.runs_dir / spec["key"]))
        loaded = _load_arm(run_dir)
        if loaded is not None:
            arms.append({**spec, **loaded})
    if not arms:
        raise SystemExit(f"no finished arms under {args.runs_dir}")
    stats_path = args.runs_dir / "data_stats.json"
    data_stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    notes_path = _HERE / "notes.html"
    notes_html = notes_path.read_text() if notes_path.exists() else ""

    extras: dict = {}
    shape_path = args.runs_dir / "shape_overhead.json"
    if shape_path.exists():
        extras["shape_overhead"] = json.loads(shape_path.read_text())
    two_pass = _load_arm(args.runs_dir / "pad_batchmax_2pass")
    if two_pass is not None:
        extras["two_pass"] = two_pass
    oom_log = _HERE / "logs" / "smoke_pad_fixed8192.out"
    if oom_log.exists():
        oom_lines = [line for line in oom_log.read_text(errors="replace").splitlines() if "OutOfMemoryError" in line]
        if oom_lines:
            extras["oom"] = oom_lines[0].split("OutOfMemoryError:", 1)[-1].strip()[:240]
    payload, body = build_payload_and_body(arms, data_stats, notes_html, extras)
    template = (_HERE / "report_template.html").read_text()
    payload_json = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    page = (
        template.replace("{{TITLE}}", "derisk-v1: padding vs MFU, Nemotron 3 Nano on Tulu-3")
        .replace("{{BODY_HTML}}", body)
        .replace("{{PAYLOAD_JSON}}", payload_json)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    logger.info("wrote %s (%.0f KiB)", args.output, len(page) / 1024)


if __name__ == "__main__":
    main()
