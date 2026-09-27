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

"""EP-imbalance and step-time model for pad-to-longest micro-batches, calibrated on derisk-v1/v2 runs.

Nemotron V3 does not pass its padding mask to the MoE, so pad tokens are routed and computed like real tokens.
In one dispatch (the same micro-batch index on all ``ep`` ranks) with padded tokens ``T`` and pad fraction ``f``,
MoE layer ``l`` gives its busiest GPU

    I_l(f) = ep * max_g [(1 - f) * real_share[l, g] + f * pad_share[l, g]]

times the mean per-GPU load, where ``real_share`` and ``pad_share`` are the fractions of real-token and
pad-token top-k assignments that land on GPU ``g``. They were measured in derisk on Tulu-3, with pads either all
``<|im_end|>`` ("eos") or random token ids ("random"); a third kind, "text_like", reuses the real-token shares
(pads the router sees in context, like derisk-v2.3b's visible natural filler). A step's time is modeled as

    step_time = A + B * N + K * sum_{dispatch d, layer l} (top_k / ep) * T_d * (I_{d,l} - 1)

where ``N`` is the padded tokens per step and the sum is the extra expert assignments on the busiest GPU. ``K``
and the balanced time at 8,192-token padding come from the three pad-fixed8192 runs (baseline, random pads,
expert placement), which share shapes but differ in measured imbalance. ``A`` and ``B`` then follow from that
balanced time and the warm (second-pass) pad-batchmax runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EP = 8
TOP_K = 6

_FIXED_RUNS = {
    "baseline": "derisk-v1/runs/pad_fixed8192",
    "random": "derisk-v2/runs/v21_pad_fixed8192_randpad",
    "placement": "derisk-v2/runs/v22_pad_fixed8192_placement",
}
_BATCHMAX_RUNS = {
    "baseline": "derisk-v1/runs/pad_batchmax_2pass",
    "random": "derisk-v2/runs/v21_pad_batchmax_randpad_2pass",
    "placement": "derisk-v2/runs/v22_pad_batchmax_placement_2pass",
}
_FIXED_STEPS = range(2, 20)  # steps 0-1 are warmup
_BATCHMAX_WARM_STEPS = range(20, 40)  # second pass over the same data: shape caches are warm
# Warm pad-batchmax runs not used for fitting, with the pad kind whose prediction they test.
_HOLDOUT_RUNS = {
    "v2.1 random pad ids": ("derisk-v2/runs/v21_pad_batchmax_randpad_2pass", "random"),
    "v2.3a natural filler, masked": ("derisk-v2/runs/v23a_pad_batchmax_natural_masked_2pass", "random"),
    "v2.3b natural filler, visible": ("derisk-v2/runs/v23b_pad_batchmax_natural_visible_2pass", "text_like"),
}


@dataclass(frozen=True)
class Calibration:
    """Routing shares and timing constants of the model; see the module docstring."""

    real_share: np.ndarray
    """Array of shape [layers, ep]: fraction of real-token assignments per GPU."""
    pad_share: dict[str, np.ndarray]
    """``"eos"`` / ``"random"`` / ``"text_like"`` -> array of shape [layers, ep]: fraction of pad-token assignments
    per GPU. ``"text_like"`` reuses the real-token shares: pads that the router sees in context, like derisk-v2.3b's
    visible natural filler, which measured 1.15x imbalance at 91% padding."""
    k_seconds: float
    """Seconds per extra expert assignment on the busiest GPU (forward, backward and recompute)."""
    a_seconds: float
    """Balanced step time at zero tokens."""
    b_seconds_per_token: float
    """Balanced step time per padded token."""
    diagnostics: dict
    """Per-run measured vs fitted numbers, for the report."""


def _step_times(run_dir: Path, steps: range) -> np.ndarray:
    records = {r["step"]: r for r in map(json.loads, (run_dir / "probe_steps.jsonl").read_text().splitlines())}
    return np.array([records[s]["probe/step_time_wall_s"] for s in steps])


def measured_run(run_dir: Path, steps: range) -> dict[str, np.ndarray]:
    """Per-step measured routing excess, padded tokens, mean imbalance and wall time of one derisk run.

    Args:
        run_dir: A derisk run directory with ``routing.npz`` and ``probe_steps.jsonl``.
        steps: Optimizer steps to use.

    Returns:
        ``excess``, ``padded_tokens``, ``imbalance`` and ``step_time``, each an array of shape [steps].
        ``excess`` is the sum over micro-batches and MoE layers of (busiest-GPU minus mean-GPU assignments);
        ``imbalance`` is the mean over micro-batches and layers of busiest / mean.
    """
    routing = np.load(run_dir / "routing.npz")
    index = [int(np.flatnonzero(routing["steps"] == s)[0]) for s in steps]
    counts = routing["counts"][index].sum(axis=3).astype(np.float64)  # [steps, micro, layers, experts]
    loads = counts.reshape(*counts.shape[:3], EP, -1).sum(axis=-1)  # [steps, micro, layers, ep]
    shapes = routing["microbatch_shapes"][index]  # [steps, ranks, micro, (batch, padded length, real tokens)]
    return {
        "excess": (loads.max(axis=-1) - loads.mean(axis=-1)).sum(axis=(1, 2)),
        "padded_tokens": (shapes[..., 0] * shapes[..., 1]).sum(axis=(1, 2)).astype(np.float64),
        "imbalance": (loads.max(axis=-1) / loads.mean(axis=-1)).mean(axis=(1, 2)),
        "step_time": _step_times(run_dir, steps),
    }


def _gpu_shares(run_dir: Path, *, pad: bool) -> np.ndarray:
    counts = np.load(run_dir / "routing.npz")["counts"].astype(np.float64)  # [steps, micro, layers, cats, experts]
    per_expert = counts[:, :, :, 1:, :].sum(axis=(0, 1, 3)) if pad else counts[:, :, :, 0, :].sum(axis=(0, 1))
    per_gpu = per_expert.reshape(per_expert.shape[0], EP, -1).sum(axis=-1)
    return per_gpu / per_gpu.sum(axis=-1, keepdims=True)


def fit_calibration(experiments_dir: Path) -> Calibration:
    """Fit the model from the derisk-v1/v2 runs under ``experiments_dir``.

    Args:
        experiments_dir: The ``experiments/`` directory that holds ``derisk-v1`` and ``derisk-v2``.

    Returns:
        The fitted :class:`Calibration`.
    """
    fixed = {k: measured_run(experiments_dir / p, _FIXED_STEPS) for k, p in _FIXED_RUNS.items()}
    batchmax = {k: measured_run(experiments_dir / p, _BATCHMAX_WARM_STEPS) for k, p in _BATCHMAX_RUNS.items()}

    excess = np.array([m["excess"].mean() for m in fixed.values()])
    times = np.array([m["step_time"].mean() for m in fixed.values()])
    k_seconds, balanced_fixed = np.polyfit(excess, times, 1)
    n_fixed = float(np.mean([m["padded_tokens"].mean() for m in fixed.values()]))
    balanced_batchmax = float(np.mean([(m["step_time"] - k_seconds * m["excess"]).mean() for m in batchmax.values()]))
    n_batchmax = float(np.mean([m["padded_tokens"].mean() for m in batchmax.values()]))
    b = (balanced_fixed - balanced_batchmax) / (n_fixed - n_batchmax)
    a = balanced_batchmax - b * n_batchmax

    diagnostics = {}
    for regime, runs in (("fixed8192", fixed), ("batchmax_warm", batchmax)):
        for key, m in runs.items():
            fitted = a + b * m["padded_tokens"].mean() + k_seconds * m["excess"].mean()
            diagnostics[f"{regime}/{key}"] = {
                "step_time_measured": float(m["step_time"].mean()),
                "step_time_sem": float(m["step_time"].std(ddof=1) / np.sqrt(len(m["step_time"]))),
                "step_time_fitted": float(fitted),
                "imbalance_measured": float(m["imbalance"].mean()),
                "padded_tokens_per_step": float(m["padded_tokens"].mean()),
            }
    return Calibration(
        real_share=_gpu_shares(experiments_dir / _FIXED_RUNS["baseline"], pad=False),
        pad_share={
            "eos": _gpu_shares(experiments_dir / _FIXED_RUNS["baseline"], pad=True),
            "random": _gpu_shares(experiments_dir / _FIXED_RUNS["random"], pad=True),
            "text_like": _gpu_shares(experiments_dir / _FIXED_RUNS["baseline"], pad=False),
        },
        k_seconds=float(k_seconds),
        a_seconds=float(a),
        b_seconds_per_token=float(b),
        diagnostics=diagnostics,
    )


def replay_sampler(
    lengths: np.ndarray, *, world: int, local_batch: int, global_batch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Replay derisk's data order and pad-to-longest collation on a list of sequence lengths.

    ``StatefulDistributedSampler`` with ``shuffle=False`` gives rank ``r`` the samples ``r, r + world, ...``;
    each rank groups consecutive ``local_batch`` of them into a micro-batch padded to its longest sequence, and
    an optimizer step takes ``global_batch / (world * local_batch)`` micro-batches per rank. Trailing samples
    that do not fill a whole step are dropped.

    Args:
        lengths: Array of shape [samples]: token count of each sample in data order (after truncation).
        world: Number of data-parallel ranks.
        local_batch: Sequences per micro-batch per rank.
        global_batch: Sequences per optimizer step.

    Returns:
        ``(real, padded)``, each an int64 array of shape [steps, micro_batches_per_step]: real and padded token
        counts of each dispatch, summed over the ranks.

    Raises:
        ValueError: If ``global_batch`` is not a multiple of ``world * local_batch`` or there is less than one step.
    """
    if global_batch % (world * local_batch):
        raise ValueError(f"global_batch {global_batch} is not a multiple of world * local_batch")
    n = (len(lengths) // global_batch) * global_batch
    if n == 0:
        raise ValueError(f"{len(lengths)} samples do not fill one step of {global_batch}")
    per_rank = np.stack([np.asarray(lengths[r:n:world], dtype=np.int64) for r in range(world)])
    micro = per_rank.reshape(world, -1, local_batch)  # [world, micro-batches, local_batch]
    real = micro.sum(axis=-1).sum(axis=0)
    padded = (micro.max(axis=-1) * local_batch).sum(axis=0)
    per_step = global_batch // (world * local_batch)
    return real.reshape(-1, per_step), padded.reshape(-1, per_step)


def predict(calibration: Calibration, real: np.ndarray, padded: np.ndarray) -> dict[str, dict[str, float]]:
    """Predict EP imbalance and step time for each pad kind in ``calibration.pad_share``.

    Args:
        calibration: Output of :func:`fit_calibration`.
        real: Array of shape [steps, micro_batches_per_step]: real tokens per dispatch (from :func:`replay_sampler`).
        padded: Array of the same shape: padded tokens per dispatch.

    Returns:
        ``{kind: {...}}`` for ``"eos"``, ``"random"`` and ``"text_like"``, each with ``imbalance`` (mean over dispatches and layers of busiest / mean
        GPU load), ``step_time`` (seconds, mean over steps) and ``real_tokens_per_s_per_gpu``.
    """
    real = real.astype(np.float64)
    padded = padded.astype(np.float64)
    pad_fraction = 1.0 - real / padded
    mean_load = TOP_K / EP * padded  # assignments per GPU per layer, [steps, micro]
    f = pad_fraction[..., None, None]
    out = {}
    for kind, pad_share in calibration.pad_share.items():
        imbalance = EP * np.max((1.0 - f) * calibration.real_share + f * pad_share, axis=-1)  # [steps, micro, layers]
        excess = (mean_load[..., None] * (imbalance - 1.0)).sum(axis=(1, 2))
        step_time = calibration.a_seconds + calibration.b_seconds_per_token * padded.sum(axis=1)
        step_time = step_time + calibration.k_seconds * excess
        out[kind] = {
            "imbalance": float(imbalance.mean()),
            "step_time": float(step_time.mean()),
            "real_tokens_per_s_per_gpu": float(real.sum(axis=1).mean() / step_time.mean() / EP),
        }
    return out


def holdout_check(experiments_dir: Path, calibration: Calibration) -> dict[str, dict[str, float]]:
    """Compare predicted and measured warm-pass savings of the derisk-v2 pad-batchmax variants (Tulu-3, 4,096).

    The baseline's own derisk-v1 data order is replayed from its recorded micro-batch shapes, so only the model is
    tested. v2.1 and v2.2 also enter the fit (through ``K`` and the balanced time); v2.3a/b do not.

    Args:
        experiments_dir: The ``experiments/`` directory that holds ``derisk-v1`` and ``derisk-v2``.
        calibration: Output of :func:`fit_calibration`.

    Returns:
        Variant label -> ``pad_kind``, ``saving_predicted`` and ``saving_measured`` (fractions of the baseline step
        time) and ``saving_measured_sem`` (standard error of the per-step paired difference, same fraction).
    """
    baseline_dir = experiments_dir / _BATCHMAX_RUNS["baseline"]
    routing = np.load(baseline_dir / "routing.npz")
    index = [int(np.flatnonzero(routing["steps"] == s)[0]) for s in _BATCHMAX_WARM_STEPS]
    shapes = routing["microbatch_shapes"][index]  # [steps, ranks, micro, (batch, padded length, real tokens)]
    padded = (shapes[..., 0] * shapes[..., 1]).sum(axis=1)
    real = shapes[..., 2].sum(axis=1)
    prediction = predict(calibration, real, padded)
    baseline = measured_run(baseline_dir, _BATCHMAX_WARM_STEPS)["step_time"]
    out = {}
    for label, (path, kind) in _HOLDOUT_RUNS.items():
        variant = measured_run(experiments_dir / path, _BATCHMAX_WARM_STEPS)["step_time"]
        # Both runs see the same data in the same order, so compare step by step (paired differences).
        sem = (baseline - variant).std(ddof=1) / np.sqrt(len(baseline))
        out[label] = {
            "pad_kind": kind,
            "saving_predicted": 1.0 - prediction[kind]["step_time"] / prediction["eos"]["step_time"],
            "saving_measured": 1.0 - variant.mean() / baseline.mean(),
            "saving_measured_sem": float(sem / baseline.mean()),
        }
    return out
