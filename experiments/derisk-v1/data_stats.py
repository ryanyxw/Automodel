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

"""Token-length statistics of the derisk-v1 subset and the padding each arm adds (CPU only).

Builds the training dataset the way the recipe does (``NeMoAutoTokenizer`` + ``ChatDatasetConfig`` from
the arm YAML), then replays the deterministic sampler (``StatefulDistributedSampler`` with
``shuffle=False``, ``drop_last=True``: rank ``r`` reads samples ``r, r + W, r + 2W, ...``) to predict
every rank's micro-batch shapes under each arm's padding rule.

Usage::

    python experiments/derisk-v1/data_stats.py --output experiments/derisk-v1/runs/data_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import yaml

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_ARMS = {
    "pad_batchmax": _HERE / "configs" / "pad_batchmax.yaml",
    "pad_fixed8192": _HERE / "configs" / "pad_fixed8192.yaml",
}


def _padded_length(max_length: int, divisible: int | None) -> int:
    """Return the collater's padded sequence length for a micro-batch (``default_collater`` rule)."""
    return max_length if not divisible else math.ceil(max_length / divisible) * divisible


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def main() -> None:
    """Tokenize the subset, replay the sampler for both arms, and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    configs = {arm: yaml.safe_load(path.read_text()) for arm, path in _ARMS.items()}
    reference = configs["pad_batchmax"]
    for arm, cfg in configs.items():
        if cfg["dataset"] != reference["dataset"] or cfg["step_scheduler"] != reference["step_scheduler"]:
            raise ValueError(f"{arm} must use the same dataset and batch sizes as pad_batchmax")

    dataset_fields = {key: value for key, value in reference["dataset"].items() if key != "_target_"}
    model_name = reference["model"]["pretrained_model_name_or_path"]
    tokenizer = NeMoAutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = ChatDatasetConfig(**dataset_fields).build(tokenizer=tokenizer)

    lengths, real_lengths, label_tokens = [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        lengths.append(len(sample["input_ids"]))
        real_lengths.append(int(sum(sample["attention_mask"])))
        label_tokens.append(sum(1 for label in sample["labels"] if label != -100))
    lengths_np, real_np, labels_np = np.asarray(lengths), np.asarray(real_lengths), np.asarray(label_tokens)

    world = args.world_size
    local_batch = reference["step_scheduler"]["local_batch_size"]
    grad_accum = reference["step_scheduler"]["global_batch_size"] // (local_batch * world)
    per_rank = len(dataset) // world
    rank_indices = [list(range(rank, per_rank * world, world)) for rank in range(world)]
    n_micro_per_rank = per_rank // local_batch

    arms = {}
    for arm, cfg in configs.items():
        divisible = cfg["dataloader"]["collate_fn"].get("pad_seq_len_divisible")
        steps = []
        for step in range(n_micro_per_rank // grad_accum):
            padded = real = 0
            micro_lengths = []
            for rank in range(world):
                for micro in range(step * grad_accum, (step + 1) * grad_accum):
                    ids = rank_indices[rank][micro * local_batch : (micro + 1) * local_batch]
                    seq = _padded_length(int(lengths_np[ids].max()), divisible)
                    micro_lengths.append(seq)
                    padded += seq * len(ids)
                    real += int(real_np[ids].sum())
            steps.append(
                {
                    "step": step,
                    "padded_tokens": padded,
                    "real_tokens": real,
                    "pad_fraction": 1.0 - real / padded,
                    "microbatch_seq_len_mean": float(np.mean(micro_lengths)),
                    "microbatch_seq_len_max": int(np.max(micro_lengths)),
                }
            )
        total_padded = sum(s["padded_tokens"] for s in steps)
        total_real = sum(s["real_tokens"] for s in steps)
        arms[arm] = {
            "pad_seq_len_divisible": divisible,
            "total_padded_tokens": total_padded,
            "total_real_tokens": total_real,
            "pad_fraction": 1.0 - total_real / total_padded,
            "steps": steps,
        }

    seq_length = reference["dataset"].get("seq_length")
    report = {
        "dataset": reference["dataset"],
        "tokenizer": model_name,
        # ChatDataset pads with its own pad id (the collater reads it from each sample), which can differ
        # from tokenizer.pad_token_id.
        "dataset_pad_token_id": dataset.pad_token_id,
        "dataset_pad_token": tokenizer.convert_ids_to_tokens(dataset.pad_token_id),
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "eos_token": tokenizer.eos_token,
        "n_samples": len(dataset),
        "world_size": world,
        "local_batch_size": local_batch,
        "grad_accum_steps": grad_accum,
        "sequence_length": _percentiles(lengths_np),
        "real_length": _percentiles(real_np),
        "label_tokens": _percentiles(labels_np),
        "samples_without_labels": int((labels_np == 0).sum()),
        "samples_at_truncation_limit": int((lengths_np >= (seq_length or 0) - 1).sum()) if seq_length else 0,
        "sequence_length_histogram": dict(
            zip(
                ["<256", "256-512", "512-1k", "1k-2k", "2k-4k", ">=4k"],
                np.histogram(lengths_np, bins=[0, 256, 512, 1024, 2048, 4096, 10**9])[0].tolist(),
            )
        ),
        "lengths": lengths,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    brief_keys = ("n_samples", "dataset_pad_token", "dataset_pad_token_id", "tokenizer_pad_token_id", "eos_token_id")
    brief = {key: report[key] for key in (*brief_keys, "sequence_length", "real_length", "label_tokens")}
    brief["samples_at_truncation_limit"] = report["samples_at_truncation_limit"]
    brief["sequence_length_histogram"] = report["sequence_length_histogram"]
    brief["arms"] = {arm: {k: v for k, v in info.items() if k != "steps"} for arm, info in arms.items()}
    logger.info("data stats: %s", json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
