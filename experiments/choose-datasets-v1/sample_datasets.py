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

"""Draw a random sample of each candidate SFT dataset from the Hugging Face Hub (choose-datasets-v1, phase 1).

For each candidate the script reads the Hub parquet files directly through ``HfFileSystem``:

1. the small metadata columns of every row (reasoning on/off, source, language, ...), to record the
   dataset-wide composition and how the files are ordered;
2. the full rows of randomly chosen row groups (drawn with probability proportional to their row count),
   keeping a random subset of rows from each, so the ordering of any one file cannot dominate the sample.

Rows are converted to OpenAI chat format (``messages``) and written in random order to
``<output-dir>/<name>.jsonl``, with ``<output-dir>/<name>.meta.json`` describing what was read.

Usage::

    source experiments/derisk-v1/env.sh
    python experiments/choose-datasets-v1/sample_datasets.py --output-dir experiments/choose-datasets-v1/samples
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

logger = logging.getLogger(__name__)


def _ifchat_row(row: dict) -> dict | None:
    """Nemotron IF-Chat-v1: keep role, content and (when present) the separate reasoning trace."""
    messages = []
    for message in row["messages"]:
        out = {"role": message["role"], "content": message["content"] or ""}
        if message.get("reasoning_content"):
            out["reasoning_content"] = message["reasoning_content"]
        messages.append(out)
    return {"messages": messages, "reasoning": row["reasoning"], "capability_target": row["capability_target"]}


def _mot_row(row: dict) -> dict | None:
    """Mixture-of-Thoughts: already OpenAI messages (reasoning inline as ``<think>``)."""
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    return {"messages": messages, "source": row["source"], "num_tokens": row["num_tokens"]}


def _wildchat_row(row: dict) -> dict | None:
    """WildChat-1M: drop conversations flagged toxic and the per-turn metadata (IP hash, headers, ...)."""
    if row["toxic"]:
        return None
    messages = [{"role": turn["role"], "content": turn["content"] or ""} for turn in row["conversation"]]
    return {"messages": messages, "language": row["language"], "model": row["model"], "turn": row["turn"]}


@dataclass(frozen=True)
class Candidate:
    """One Hub dataset split to sample."""

    name: str
    """Output file stem."""
    repo: str
    """``HfFileSystem`` root, ``datasets/<org>/<name>`` with an optional ``@<revision>``."""
    file_filter: str
    """Substring that every selected parquet path must contain."""
    meta_columns: tuple[str, ...]
    """Small columns read for every row, to describe the whole split."""
    row_columns: tuple[str, ...]
    """Columns read for the sampled row groups; ``to_row`` consumes them."""
    to_row: Callable[[dict], dict | None]
    """Converts one Hub row to a JSONL row with ``messages``; returning ``None`` drops the row."""
    n_row_groups: int
    """Row groups to draw."""
    rows_per_group: int
    """Rows kept from each drawn row group (after ``to_row`` filtering)."""


CANDIDATES = (
    Candidate(
        name="nemotron_ifchat_v1",
        # The original split is one 7 GB JSONL; the Hub's parquet export ("partial") holds its first ~74% of rows.
        repo="datasets/nvidia/Nemotron-Instruction-Following-Chat-v1@refs%2Fconvert%2Fparquet",
        file_filter="/partial-chat_if/",
        meta_columns=("reasoning", "capability_target"),
        row_columns=("messages", "reasoning", "capability_target"),
        to_row=_ifchat_row,
        n_row_groups=40,
        rows_per_group=150,
    ),
    Candidate(
        name="mixture_of_thoughts",
        repo="datasets/open-r1/Mixture-of-Thoughts",
        file_filter="/all/",
        meta_columns=("source", "num_tokens"),
        row_columns=("messages", "source", "num_tokens"),
        to_row=_mot_row,
        n_row_groups=30,
        rows_per_group=200,
    ),
    Candidate(
        name="wildchat_1m",
        repo="datasets/allenai/WildChat-1M",
        file_filter="/data/",
        meta_columns=("model", "language", "turn", "toxic"),
        row_columns=("conversation", "toxic", "language", "turn", "model"),
        to_row=_wildchat_row,
        n_row_groups=40,
        rows_per_group=150,
    ),
)


def _is_categorical(values: list) -> bool:
    non_null = [v for v in values if v is not None]
    return not non_null or isinstance(non_null[0], (bool, str))


def _summarize_column(values: list) -> dict:
    """Value counts for categorical columns, percentiles for numeric ones."""
    if _is_categorical(values):
        return {"counts": {str(k): n for k, n in collections.Counter(values).most_common(20)}}
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return {
        "mean": float(array.mean()),
        **{f"p{q}": float(np.percentile(array, q)) for q in (10, 50, 90, 99)},
        "max": float(array.max()),
    }


def _read_meta(fs: HfFileSystem, path: str, columns: tuple[str, ...]) -> tuple[list[int], dict[str, list]]:
    """Read ``columns`` for every row of one parquet file, plus its row-group sizes."""
    parquet = pq.ParquetFile(path, filesystem=fs, pre_buffer=True)
    sizes = [parquet.metadata.row_group(g).num_rows for g in range(parquet.metadata.num_row_groups)]
    table = parquet.read(columns=list(columns))
    return sizes, {column: table.column(column).to_pylist() for column in columns}


def _read_rows(fs: HfFileSystem, path: str, group: int, columns: tuple[str, ...]) -> tuple[list[dict], int]:
    """Read one row group; return its rows and the compressed bytes of the columns read."""
    parquet = pq.ParquetFile(path, filesystem=fs, pre_buffer=True)
    meta = parquet.metadata.row_group(group)
    nbytes = sum(
        meta.column(i).total_compressed_size
        for i in range(meta.num_columns)
        if meta.column(i).path_in_schema.split(".")[0] in columns
    )
    return parquet.read_row_group(group, columns=list(columns)).to_pylist(), nbytes


def sample_candidate(
    candidate: Candidate, fs: HfFileSystem, rng: np.random.Generator, workers: int
) -> tuple[list, dict]:
    """Scan one candidate's metadata columns, draw row groups, and return (JSONL rows, metadata record)."""
    paths = sorted(p for p in fs.find(candidate.repo) if p.endswith(".parquet") and candidate.file_filter in p)
    if not paths:
        raise FileNotFoundError(f"no parquet files under {candidate.repo} matching {candidate.file_filter!r}")
    with ThreadPoolExecutor(workers) as pool:
        metas = list(pool.map(lambda p: _read_meta(fs, p, candidate.meta_columns), paths))

    groups = [(path, g, size) for path, (sizes, _) in zip(paths, metas) for g, size in enumerate(sizes)]
    population = {c: sum((cols[c] for _, cols in metas), []) for c in candidate.meta_columns}
    # How ordered is the split? Share of the most common value of the first categorical column, per row group.
    spread = {}
    first_cat = next((c for c in candidate.meta_columns if _is_categorical(population[c])), None)
    if first_cat is not None:
        top = collections.Counter(population[first_cat]).most_common(1)[0][0]
        shares, start = [], 0
        for _, _, size in groups:
            chunk = population[first_cat][start : start + size]
            shares.append(float(np.mean([v == top for v in chunk])))
            start += size
        spread = {
            "column": first_cat,
            "value": top,
            "share_min": min(shares),
            "share_median": float(np.median(shares)),
            "share_max": max(shares),
        }

    weights = np.array([size for *_, size in groups], dtype=np.float64)
    chosen = rng.choice(len(groups), size=candidate.n_row_groups, replace=False, p=weights / weights.sum())
    with ThreadPoolExecutor(workers) as pool:
        reads = list(pool.map(lambda i: _read_rows(fs, groups[i][0], groups[i][1], candidate.row_columns), chosen))

    rows, drawn = [], []
    for i, (group_rows, nbytes) in zip(chosen, reads):
        converted = [r for r in (candidate.to_row(row) for row in group_rows) if r is not None]
        keep = rng.choice(len(converted), size=min(candidate.rows_per_group, len(converted)), replace=False)
        rows.extend(converted[k] for k in sorted(keep))
        path, group, size = groups[i]
        drawn.append({"file": path, "row_group": group, "rows": size, "kept": len(keep), "bytes": nbytes})
    order = rng.permutation(len(rows))
    rows = [rows[k] for k in order]

    record = {
        "name": candidate.name,
        "repo": candidate.repo,
        "file_filter": candidate.file_filter,
        "files": len(paths),
        "rows_total": len(population[candidate.meta_columns[0]]),
        "row_groups_total": len(groups),
        "population": {c: _summarize_column(v) for c, v in population.items()},
        "row_group_spread": spread,
        "row_groups_drawn": drawn,
        "bytes_downloaded": sum(d["bytes"] for d in drawn),
        "rows_sampled": len(rows),
    }
    return rows, record


def main() -> None:
    """Sample every candidate and write ``<name>.jsonl`` and ``<name>.meta.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="candidate names to sample (default: all)")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for index, candidate in enumerate(CANDIDATES):
        if args.only and candidate.name not in args.only:
            continue
        rng = np.random.default_rng([args.seed, index])
        logger.info("sampling %s", candidate.name)
        rows, record = sample_candidate(candidate, fs, rng, args.workers)
        record["seed"] = [args.seed, index]
        with (args.output_dir / f"{candidate.name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")  # ASCII-escaped: ChatDataset splits JSONL with str.splitlines()
        (args.output_dir / f"{candidate.name}.meta.json").write_text(json.dumps(record, indent=2))
        logger.info(
            "%s: %d rows from %d row groups (%.0f MB), split has %d rows; ordering: %s",
            candidate.name,
            record["rows_sampled"],
            len(record["row_groups_drawn"]),
            record["bytes_downloaded"] / 1e6,
            record["rows_total"],
            record["row_group_spread"],
        )


if __name__ == "__main__":
    main()
