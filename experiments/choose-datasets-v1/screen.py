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

"""Tokenize each sampled dataset the way the recipe does and predict what random pad tokens would save.

Inputs are ``<samples-dir>/<name>.jsonl`` from ``sample_datasets.py`` plus derisk's Tulu-3 subset (the
baseline; its first 2,560 rows are exactly the derisk-v1/v2 data in the same order). For each dataset:

1. build ``ChatDataset`` with the Nemotron 3 Nano tokenizer, untruncated and at each truncation length, with the
   derisk configs' arguments, and record every sample's token and label counts;
2. replay derisk's sampler (8 ranks, global batch 128, each micro-batch padded to its longest sequence) over
   random 2,560-sample orders, for 2 and 4 sequences per micro-batch;
3. predict EP imbalance and step time with ``<|im_end|>`` pads and with random pads (``ep_model``).

Usage::

    source experiments/derisk-v1/env.sh
    python experiments/choose-datasets-v1/screen.py --samples-dir experiments/choose-datasets-v1/samples \
        --output-dir experiments/choose-datasets-v1/runs
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import multiprocessing
import os
import zlib
from pathlib import Path

import ep_model
import numpy as np

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset, ChatDatasetConfig

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CAPS = (4096, 8192)
LOCAL_BATCHES = (2, 4)
WORLD, GLOBAL_BATCH, SUBSET = 8, 128, 2560  # derisk: 20 steps x 128 sequences
TULU = {"path_or_dataset_id": "allenai/tulu-3-sft-mixture", "shuffle_seed": 42}
META_KEYS = ("reasoning", "capability_target", "source", "language", "model")
# Nano's stock template drops earlier turns' reasoning, which ChatDataset's fallback loss mask rejects for
# multi-turn reasoning rows; the same template with {% generation %} markers renders identical text
# (make_generation_template.py checks this).
TEMPLATE_OVERRIDES = {"nemotron_ifchat_v1": _HERE / "templates" / "nano3_generation.jinja"}

_worker_dataset: ChatDataset | None = None


def _count(index: int) -> tuple[int, int, str]:
    try:
        sample = _worker_dataset[index]
    except (ValueError, AssertionError) as error:  # a conversation ChatDataset rejects: reported and excluded
        return -1, -1, f"{type(error).__name__}: {error}"[:300]
    return len(sample["input_ids"]), sum(1 for label in sample["labels"] if label != -100), ""


def _token_counts(dataset: ChatDataset, workers: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Token and label counts of every sample (-1 where ``ChatDataset`` raised), plus the error messages."""
    global _worker_dataset
    _worker_dataset = dataset
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        counts = pool.map(_count, range(len(dataset)), chunksize=16)
    _worker_dataset = None
    tokens, labels, errors = zip(*counts)
    return np.asarray(tokens, dtype=np.int64), np.asarray(labels, dtype=np.int64), [e for e in errors if e]


def _build(source: dict, cap: int | None, tokenizer) -> ChatDataset:
    kwargs = {"seq_length": cap, "truncation": True} if cap else {}
    return ChatDatasetConfig(**source, padding="do_not_pad", **kwargs).build(tokenizer=tokenizer)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        **{f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 90, 99)},
        "max": float(values.max()),
    }


def _screen_order(calibration: ep_model.Calibration, lengths: np.ndarray, local_batch: int) -> dict[str, float]:
    """Replay one data order and return the batch and prediction metrics."""
    real, padded = ep_model.replay_sampler(lengths, world=WORLD, local_batch=local_batch, global_batch=GLOBAL_BATCH)
    pred = ep_model.predict(calibration, real, padded)
    per_rank = lengths[: real.size * WORLD * local_batch].reshape(-1, local_batch)  # every micro-batch's lengths
    saving = pred["eos"]["step_time"] - pred["random"]["step_time"]
    return {
        "pad_fraction": float(1.0 - real.sum() / padded.sum()),
        "tokens_per_microbatch": float(padded.mean() / WORLD),
        "pad_tokens_per_microbatch": float((padded - real).mean() / WORLD),
        "max_microbatch_tokens": int(per_rank.max(axis=1).max() * local_batch),
        "imbalance_eos": pred["eos"]["imbalance"],
        "imbalance_random": pred["random"]["imbalance"],
        "step_time_eos": pred["eos"]["step_time"],
        "step_time_random": pred["random"]["step_time"],
        "saving_s": saving,
        "saving_frac": saving / pred["eos"]["step_time"],
        "effective_mfu_gain": pred["eos"]["step_time"] / pred["random"]["step_time"] - 1.0,
        "imbalance_text_like": pred["text_like"]["imbalance"],
        "saving_frac_text_like": 1.0 - pred["text_like"]["step_time"] / pred["eos"]["step_time"],
        "real_tokens_per_s_per_gpu_eos": pred["eos"]["real_tokens_per_s_per_gpu"],
    }


def _aggregate(orders: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([o[key] for o in orders])),
            "p5": float(np.percentile([o[key] for o in orders], 5)),
            "p95": float(np.percentile([o[key] for o in orders], 95)),
        }
        for key in orders[0]
    }


def screen_dataset(
    name: str, source: dict, meta: list[dict], tokenizer, calibration: ep_model.Calibration, args: argparse.Namespace
) -> tuple[dict, dict[str, np.ndarray]]:
    """Tokenize one dataset at every truncation length and screen random 2,560-sample orders."""
    arrays: dict[str, np.ndarray] = {}
    errors: list[str] = []
    for cap in (None, *CAPS):
        tokens, labels, cap_errors = _token_counts(_build(source, cap, tokenizer), args.workers)
        arrays[f"tokens_{cap or 'full'}"], arrays[f"labels_{cap or 'full'}"] = tokens, labels
        errors.extend(cap_errors)
        logger.info("%s cap=%s: %d samples, %d rejected", name, cap, len(tokens), len(cap_errors))
    valid = np.all([arrays[f"tokens_{cap or 'full'}"] >= 0 for cap in (None, *CAPS)], axis=0)
    arrays = {key: value[valid] for key, value in arrays.items()}
    if meta:
        meta = [row for row, keep in zip(meta, valid) if keep]
    n = int(valid.sum())
    rng = np.random.default_rng([args.seed, zlib_hash(name)])
    orders = [rng.choice(n, size=SUBSET, replace=False) for _ in range(args.orders)]

    result = {
        "samples": n,
        "rejected": {"count": int((~valid).sum()), "examples": sorted(set(errors))[:3]},
        "length_full": _percentiles(arrays["tokens_full"]),
        "caps": {},
    }
    for cap in CAPS:
        tokens, labels = arrays[f"tokens_{cap}"], arrays[f"labels_{cap}"]
        entry = {
            "length": _percentiles(tokens),
            "label_tokens": _percentiles(labels),
            "truncated_share": float((arrays["tokens_full"] > tokens).mean()),
            "no_label_share": float((labels == 0).mean()),
            "screen": {},
        }
        for local_batch in LOCAL_BATCHES:
            screened = [_screen_order(calibration, tokens[order], local_batch) for order in orders]
            entry["screen"][f"lbs{local_batch}"] = _aggregate(screened)
            if name == "tulu3":  # derisk's exact order: the first 2,560 rows of the seed-42 shuffle
                entry["screen"][f"lbs{local_batch}_derisk_order"] = _screen_order(
                    calibration, tokens[:SUBSET], local_batch
                )
        result["caps"][str(cap)] = entry

    # Length by metadata category (e.g. reasoning on/off, source), at full length and at the 8,192 cap.
    breakdown = {}
    for key in META_KEYS:
        values = [row.get(key) for row in meta]
        if all(v is None for v in values):
            continue
        groups = collections.defaultdict(list)
        for i, value in enumerate(values):
            groups[str(value)].append(i)
        top = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]
        breakdown[key] = {
            value: {
                "share": len(idx) / n,
                "mean_full": float(arrays["tokens_full"][idx].mean()),
                "median_full": float(np.median(arrays["tokens_full"][idx])),
                "mean_8192": float(arrays["tokens_8192"][idx].mean()),
            }
            for value, idx in top
        }
    result["breakdown"] = breakdown
    return result, arrays


def zlib_hash(text: str) -> int:
    """Stable 32-bit hash of a dataset name, for per-dataset RNG streams."""
    return zlib.crc32(text.encode())


def main() -> None:
    """Screen the baseline and every sampled candidate; write ``screen.json`` and per-dataset length arrays."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples-dir", type=Path, default=_HERE / "samples")
    parser.add_argument("--output-dir", type=Path, default=_HERE / "runs")
    parser.add_argument("--baseline-rows", type=int, default=6000)
    parser.add_argument("--orders", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    (args.output_dir / "lengths").mkdir(parents=True, exist_ok=True)

    calibration = ep_model.fit_calibration(_HERE.parent)
    tokenizer = NeMoAutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    datasets = {"tulu3": ({**TULU, "split": f"train[:{args.baseline_rows}]"}, [])}
    for path in sorted(args.samples_dir.glob("*.jsonl")):
        meta = [{k: row.get(k) for k in META_KEYS} for row in map(json.loads, path.read_text().splitlines())]
        source = {"path_or_dataset_id": str(path)}
        if path.stem in TEMPLATE_OVERRIDES:
            source["chat_template"] = str(TEMPLATE_OVERRIDES[path.stem])
        datasets[path.stem] = (source, meta)

    report = {
        "model": MODEL,
        "world": WORLD,
        "global_batch": GLOBAL_BATCH,
        "subset": SUBSET,
        "orders": args.orders,
        "calibration": {
            "k_seconds": calibration.k_seconds,
            "a_seconds": calibration.a_seconds,
            "b_seconds_per_token": calibration.b_seconds_per_token,
            "runs": calibration.diagnostics,
            "holdout": ep_model.holdout_check(_HERE.parent, calibration),
        },
        "datasets": {},
    }
    for name, (source, meta) in datasets.items():
        result, arrays = screen_dataset(name, source, meta, tokenizer, calibration, args)
        report["datasets"][name] = result
        np.savez_compressed(args.output_dir / "lengths" / f"{name}.npz", **arrays)
        best = result["caps"]["8192"]["screen"]["lbs2"]
        logger.info(
            "%s @8192 lbs2: pad %.1f%%, %.0f tokens/micro-batch, imbalance %.2f -> %.2f, saving %.2f s (%.1f%%)",
            name,
            100 * best["pad_fraction"]["mean"],
            best["tokens_per_microbatch"]["mean"],
            best["imbalance_eos"]["mean"],
            best["imbalance_random"]["mean"],
            best["saving_s"]["mean"],
            100 * best["saving_frac"]["mean"],
        )
    (args.output_dir / "screen.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
