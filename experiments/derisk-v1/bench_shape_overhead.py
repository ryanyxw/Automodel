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

"""Measure the per-shape cost of dynamic padding on one GPU (derisk-v1 diagnostic).

Padding each micro-batch to its own longest sequence gives almost every micro-batch a sequence length
the process has not seen before. Kernels that build or tune something per shape then pay that cost
again. This times forward + backward of two such kernels on the Nemotron 3 Nano shapes, the first
time each length is seen ("cold") and again on a repeat ("warm"):

* TE ``DotProductAttention`` with the ``padding_causal`` mask the model uses (cuDNN fused attention
  builds an execution plan per shape);
* cut-cross-entropy ``linear_cross_entropy`` against the 131,072-token LM head (Triton autotuning is
  keyed on a binned token count).

Usage::

    source experiments/derisk-v1/env.sh
    python experiments/derisk-v1/bench_shape_overhead.py --output experiments/derisk-v1/runs/shape_overhead.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from cut_cross_entropy import linear_cross_entropy
from transformer_engine.pytorch import DotProductAttention

logger = logging.getLogger(__name__)

HIDDEN, VOCAB, HEADS, KV_HEADS, HEAD_DIM = 2688, 131072, 32, 2, 128


def _timed(fn: Callable[[], None]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _attention_step(attn: DotProductAttention, batch: int, seq_len: int) -> Callable[[], None]:
    """Build one TE attention forward + backward at [batch, seq_len] with the second row half padded."""

    def run() -> None:
        q = torch.randn(batch, seq_len, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(batch, seq_len, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(batch, seq_len, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        mask = torch.zeros(batch, 1, 1, seq_len, dtype=torch.bool, device="cuda")
        mask[-1, ..., seq_len // 2 :] = True  # True marks padding
        attn(q, k, v, attention_mask=mask).float().sum().backward()

    return run


def _cce_step(classifier: torch.Tensor, tokens: int) -> Callable[[], None]:
    """Build one cut-cross-entropy forward + backward over ``tokens`` hidden states."""

    def run() -> None:
        hidden = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        targets = torch.randint(0, VOCAB, (tokens,), device="cuda")
        linear_cross_entropy(hidden, classifier, targets).backward()

    return run


def main() -> None:
    """Time cold and warm calls over a list of distinct sequence lengths and write a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, default=24, help="number of distinct sequence lengths")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(0)
    lengths = sorted({int(x) for x in rng.integers(256, 4096, size=args.lengths * 2)})[: args.lengths]
    attn = DotProductAttention(
        num_attention_heads=HEADS,
        kv_channels=HEAD_DIM,
        num_gqa_groups=KV_HEADS,
        attn_mask_type="padding_causal",
        qkv_format="bshd",
    ).cuda()
    classifier = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # One call at an unrelated shape first, so one-time library initialization is not billed to shape 1.
    _attention_step(attn, args.batch, 128)()
    _cce_step(classifier, args.batch * 128)()

    results = {}
    for name, make in (
        ("te_attention", lambda s: _attention_step(attn, args.batch, s)),
        ("cut_cross_entropy", lambda s: _cce_step(classifier, args.batch * s)),
    ):
        cold = [_timed(make(s)) for s in lengths]
        warm = [_timed(make(s)) for s in lengths]
        results[name] = {
            "cold_ms": [1000 * t for t in cold],
            "warm_ms": [1000 * t for t in warm],
            "cold_mean_ms": 1000 * float(np.mean(cold)),
            "warm_mean_ms": 1000 * float(np.mean(warm)),
            "overhead_per_new_shape_ms": 1000 * float(np.mean(cold) - np.mean(warm)),
        }
        logger.info(
            "%s: cold %.1f ms, warm %.1f ms, overhead per new shape %.1f ms",
            name,
            results[name]["cold_mean_ms"],
            results[name]["warm_mean_ms"],
            results[name]["overhead_per_new_shape_ms"],
        )
    report = {"batch": args.batch, "lengths": lengths, "gpu": torch.cuda.get_device_name(0), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
