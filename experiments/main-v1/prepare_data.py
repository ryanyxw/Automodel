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

"""Build main-v1's training subset and natural-text filler from Nemotron IF-Chat v1 (CPU only).

1. Shuffle all rows of ``data/chat_if.jsonl`` from ``nvidia/Nemotron-Instruction-Following-Chat-v1`` (pinned
   revision) with a fixed seed.
2. Walk the shuffled order and keep the rows the training recipe can use: ``ChatDataset``, configured exactly
   as in ``configs/fixed_default.yaml`` (Nano chat template with generation markers, truncation to 8,192
   tokens), formats them without error, and at least one supervised (assistant) token survives truncation.
3. The first ``--train-rows`` kept rows are the training subset, written to ``data/train.jsonl`` in shuffled
   order. The dataloader does not shuffle, so every run sees this order.
4. The next kept rows, until their tokens reach ``--filler-tokens``, are formatted the same way and
   concatenated into ``data/filler.npy``: the natural-text filler of ``pad_fill.natural_pad_collater``. It
   shares no conversation with the training subset.
5. ``data/train_lengths.npz`` holds per-row token counts and ``data/data_stats.json`` what was kept and why.

Usage::

    source experiments/main-v1/env.sh
    python experiments/main-v1/prepare_data.py
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import hf_hub_download

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset, ChatDatasetConfig

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_BASE_CONFIG = _HERE / "configs" / "fixed_default.yaml"
SOURCE_REPO = "nvidia/Nemotron-Instruction-Following-Chat-v1"
SOURCE_FILE = "data/chat_if.jsonl"
SOURCE_REVISION = "83dcd3aded0d289b0bbc018d3f9af4c5dd4005df"
_ROW_FIELDS = ("uuid", "messages", "tools", "reasoning", "capability_target")

_worker_datasets: tuple[ChatDataset, ChatDataset] | None = None


def _dataset_fields() -> tuple[str, dict]:
    """Return the training model name and the ``dataset:`` fields of the base config (without ``_target_``).

    Raises:
        ValueError: If the base config does not train on ``data/train.jsonl``.
    """
    cfg = yaml.safe_load(_BASE_CONFIG.read_text())
    fields = {key: value for key, value in cfg["dataset"].items() if key != "_target_"}
    expected = str((_HERE / "data" / "train.jsonl").relative_to(_REPO_ROOT))
    if fields["path_or_dataset_id"] != expected:
        raise ValueError(f"{_BASE_CONFIG} trains on {fields['path_or_dataset_id']}, expected {expected}")
    return cfg["model"]["pretrained_model_name_or_path"], fields


def _init_worker(model: str, fields: dict, rows_path: str) -> None:
    """Build, once per worker process, the training-format dataset and an untruncated twin over ``rows_path``."""
    global _worker_datasets
    tokenizer = NeMoAutoTokenizer.from_pretrained(model, trust_remote_code=True)
    truncated = ChatDatasetConfig(**{**fields, "path_or_dataset_id": rows_path}).build(tokenizer=tokenizer)
    untruncated_fields = {**fields, "path_or_dataset_id": rows_path, "seq_length": None, "truncation": False}
    untruncated = ChatDatasetConfig(**untruncated_fields).build(tokenizer=tokenizer)
    _worker_datasets = (truncated, untruncated)


def _format_row(index: int) -> dict:
    """Format one row in training format and untruncated; report its token counts or the formatting error."""
    truncated, untruncated = _worker_datasets
    try:
        sample = truncated[index]
        full = untruncated[index]
    except (ValueError, AssertionError, KeyError, TypeError) as error:
        return {"error": f"{type(error).__name__}: {error}"[:200]}
    return {
        "input_ids": np.asarray(sample["input_ids"], dtype=np.int32),
        "tokens": len(sample["input_ids"]),
        "labels": int((np.asarray(sample["labels"]) != -100).sum()),
        "tokens_untruncated": len(full["input_ids"]),
    }


def _format_rows(rows: list[dict], model: str, fields: dict, workers: int) -> list[dict]:
    """Format ``rows`` with ``ChatDataset`` in ``workers`` processes (via a temporary ASCII-escaped JSONL)."""
    with tempfile.TemporaryDirectory() as tmp:
        rows_path = str(Path(tmp) / "rows.jsonl")
        _write_jsonl(Path(rows_path), rows)
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(model, fields, rows_path)) as pool:
            return list(pool.map(_format_row, range(len(rows)), chunksize=16))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    # ChatDataset splits JSONL files with str.splitlines(), which also breaks on U+2028 / U+2029 / U+0085
    # inside JSON strings; json.dumps' default ASCII escaping keeps every row on one line.
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        **{f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 90, 99)},
        "max": float(values.max()),
    }


def main() -> None:
    """Select the training rows and filler rows, verify them, and write the data files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-rows", type=int, default=5120)
    parser.add_argument("--filler-tokens", type=int, default=6_000_000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--chunk", type=int, default=4096, help="rows formatted per round")
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=_HERE / "data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, fields = _dataset_fields()
    source = Path(hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REVISION))
    lines = source.read_bytes().split(b"\n")
    if lines and not lines[-1].strip():
        lines.pop()
    order = np.random.default_rng(args.seed).permutation(len(lines))
    logger.info("%s: %d rows; shuffled with seed %d", source, len(lines), args.seed)

    train: list[tuple[int, dict, dict]] = []
    filler: list[tuple[int, dict, dict]] = []
    rejected: collections.Counter[str] = collections.Counter()
    examined = 0
    filler_tokens = 0
    while len(train) < args.train_rows or filler_tokens < args.filler_tokens:
        if examined >= len(order):
            raise RuntimeError(f"ran out of rows after {examined}: {len(train)} training rows, {filler_tokens} filler")
        indices = order[examined : examined + args.chunk].tolist()
        examined += len(indices)
        rows = []
        for index in indices:
            raw = json.loads(lines[index])
            rows.append({key: raw[key] for key in _ROW_FIELDS})
        for index, row, result in zip(indices, rows, _format_rows(rows, model, fields, args.workers)):
            if "error" in result:
                rejected[result["error"].split(":")[0]] += 1
            elif result["labels"] == 0:
                rejected["no supervised token after truncation"] += 1
            elif len(train) < args.train_rows:
                train.append((index, row, result))
            elif filler_tokens < args.filler_tokens:
                filler.append((index, row, result))
                filler_tokens += result["tokens"]
        logger.info(
            "examined %d rows: %d training, %d filler (%d tokens)", examined, len(train), len(filler), filler_tokens
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    _write_jsonl(train_path, [row for _, row, _ in train])
    _write_jsonl(args.output_dir / "filler_source.jsonl", [row for _, row, _ in filler])
    stream = np.concatenate([result["input_ids"] for _, _, result in filler])
    np.save(args.output_dir / "filler.npy", stream)

    # The training runs read train.jsonl itself; check that it formats exactly as the selection did.
    written = [json.loads(line) for line in train_path.read_text().splitlines()]
    check = _format_rows(written, model, fields, args.workers)
    for position, ((_, _, selected), again) in enumerate(zip(train, check)):
        if "error" in again or not np.array_equal(selected["input_ids"], again["input_ids"]):
            raise RuntimeError(f"row {position} of {train_path} formats differently when read back: {again}")

    tokens = np.asarray([result["tokens"] for _, _, result in train])
    labels = np.asarray([result["labels"] for _, _, result in train])
    untruncated = np.asarray([result["tokens_untruncated"] for _, _, result in train])
    np.savez(
        args.output_dir / "train_lengths.npz",
        source_row=np.asarray([index for index, _, _ in train]),
        tokens=tokens,
        labels=labels,
        tokens_untruncated=untruncated,
    )
    seq_length = int(fields["seq_length"])
    stats = {
        "source": {"repo": SOURCE_REPO, "file": SOURCE_FILE, "revision": SOURCE_REVISION, "rows": len(lines)},
        "seed": args.seed,
        "rows_examined": examined,
        "rows_rejected": dict(rejected),
        "dataset_fields": fields,
        "train": {
            "rows": len(train),
            "reasoning": dict(collections.Counter(row["reasoning"] for _, row, _ in train)),
            "capability_target": dict(collections.Counter(row["capability_target"] for _, row, _ in train)),
            "turns_user": _percentiles(
                np.asarray([sum(m["role"] == "user" for m in row["messages"]) for _, row, _ in train], dtype=float)
            ),
            "tokens": _percentiles(tokens.astype(float)),
            "tokens_untruncated": _percentiles(untruncated.astype(float)),
            "label_tokens": _percentiles(labels.astype(float)),
            "truncated_share": float((untruncated > tokens).mean()),
            "seq_length": seq_length,
            "total_tokens": int(tokens.sum()),
        },
        "filler": {"rows": len(filler), "tokens": int(stream.size), "distinct_ids": int(np.unique(stream).size)},
    }
    (args.output_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("wrote %s", json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
