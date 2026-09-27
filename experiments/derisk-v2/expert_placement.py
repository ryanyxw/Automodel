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

"""derisk-v2.2: per-layer expert placement that evens out the routed-expert work per GPU.

With expert parallelism, expert ``e`` of a layer lives on EP rank ``e // (n_experts // ep_size)``, and every
dispatch waits for the busiest rank. In derisk-v1 all pad tokens of a layer went to the same few experts,
so the ranks hosting them were stragglers. This module:

* ``compute_placement`` chooses, for every MoE layer, which experts each EP rank hosts, to minimize the
  mean over dispatches of (busiest rank's assignments / mean assignments), using derisk-v1's recorded
  per-dispatch router counts. Greedy longest-processing-time assignment, then pairwise swaps.
* ``apply_expert_placement`` relabels the experts of a built, sharded model in place: expert ``e`` becomes
  expert ``old_to_new[e]``, and the router weight rows and score-correction bias entries move with it,
  so the model computes exactly the same function; only which GPU hosts which expert changes.

Usage (offline, CPU)::

    python experiments/derisk-v2/expert_placement.py \
        --routing experiments/derisk-v1/runs/pad_fixed8192/routing.npz \
        --output experiments/derisk-v2/placements/pad_fixed8192.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from nemo_automodel.components.moe.layers import MoE

logger = logging.getLogger(__name__)

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def dispatch_imbalance(rank_loads: np.ndarray) -> np.ndarray:
    """Busiest rank's load divided by the mean rank load, per dispatch.

    Args:
        rank_loads: Array of shape [dispatches, ep_ranks] with the routed assignments each rank receives.

    Returns:
        Array of shape [dispatches]; 1.0 is perfectly balanced, ``ep_ranks`` means one rank gets everything.
    """
    return rank_loads.max(axis=1) / np.maximum(rank_loads.mean(axis=1), 1e-9)


def compute_layer_placement(
    counts: np.ndarray, ep_size: int, *, iterations: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Assign one layer's experts to EP ranks (equal count per rank) to even out per-dispatch load.

    Args:
        counts: Array of shape [dispatches, experts] with the top-k assignments each expert received in
            each dispatch (all token categories summed).
        ep_size: Number of EP ranks; must divide the number of experts.
        iterations: Random pairwise swaps tried after the greedy start.
        rng: Random generator for the swap proposals.

    Returns:
        ``rank_of_expert`` (int64 array of shape [experts]) and the resulting mean dispatch imbalance.
    """
    n_dispatch, n_experts = counts.shape
    per_rank = n_experts // ep_size
    rank_of_expert = np.full(n_experts, -1, dtype=np.int64)
    rank_total = np.zeros(ep_size)
    rank_size = np.zeros(ep_size, dtype=np.int64)
    for expert in np.argsort(-counts.sum(axis=0), kind="stable"):
        free = np.flatnonzero(rank_size < per_rank)
        rank = free[np.argmin(rank_total[free])]
        rank_of_expert[expert] = rank
        rank_total[rank] += counts[:, expert].sum()
        rank_size[rank] += 1
    rank_loads = np.zeros((n_dispatch, ep_size))
    for rank in range(ep_size):
        rank_loads[:, rank] = counts[:, rank_of_expert == rank].sum(axis=1)
    best = float(dispatch_imbalance(rank_loads).mean())
    for _ in range(iterations):
        a, b = rng.integers(n_experts, size=2)
        rank_a, rank_b = rank_of_expert[a], rank_of_expert[b]
        if rank_a == rank_b:
            continue
        moved = counts[:, a] - counts[:, b]
        rank_loads[:, rank_a] -= moved
        rank_loads[:, rank_b] += moved
        value = float(dispatch_imbalance(rank_loads).mean())
        if value < best - 1e-12:
            best = value
            rank_of_expert[a], rank_of_expert[b] = rank_b, rank_a
        else:
            rank_loads[:, rank_a] += moved
            rank_loads[:, rank_b] -= moved
    return rank_of_expert, best


def compute_placement(routing_path: Path, *, iterations: int = 20000, seed: int = 0) -> dict:
    """Compute a per-layer placement from a derisk-v1 ``routing.npz``.

    Args:
        routing_path: ``routing.npz`` written by derisk-v1's RoutingProbe (``counts`` of shape [steps,
            microbatches, layers, token categories, experts]).
        iterations: Swap proposals per layer.
        seed: Seed for the swap proposals.

    Returns:
        A JSON-serializable dict with, per layer id, ``new_to_old`` (expert ids in their new order: new
        index ``j`` holds old expert ``new_to_old[j]``, so EP rank ``r`` hosts old experts
        ``new_to_old[r * per_rank:(r + 1) * per_rank]``) and the baseline / predicted mean and max
        dispatch imbalance on the derisk-v1 counts.
    """
    data = np.load(routing_path, allow_pickle=False)
    meta = json.loads(str(data["metadata"]))
    counts = data["counts"].astype(np.float64).sum(axis=3)  # [steps, microbatches, layers, experts]
    n_steps, n_micro, n_layers, n_experts = counts.shape
    ep_size = int(meta["ep_size"])
    per_rank = n_experts // ep_size
    rng = np.random.default_rng(seed)
    layers = {}
    for slot, layer_id in enumerate(data["layer_ids"].tolist()):
        layer_counts = counts[:, :, slot, :].reshape(n_steps * n_micro, n_experts)
        baseline = dispatch_imbalance(layer_counts.reshape(-1, ep_size, per_rank).sum(axis=2))
        rank_of_expert, _ = compute_layer_placement(layer_counts, ep_size, iterations=iterations, rng=rng)
        new_to_old = np.concatenate([np.flatnonzero(rank_of_expert == r) for r in range(ep_size)])
        predicted = dispatch_imbalance(layer_counts[:, new_to_old].reshape(-1, ep_size, per_rank).sum(axis=2))
        layers[str(layer_id)] = {
            "new_to_old": new_to_old.tolist(),
            "baseline_imbalance_mean": float(baseline.mean()),
            "baseline_imbalance_max": float(baseline.max()),
            "predicted_imbalance_mean": float(predicted.mean()),
            "predicted_imbalance_max": float(predicted.max()),
        }
    return {"source": str(routing_path), "ep_size": ep_size, "n_experts": n_experts, "layers": layers}


def _permute_expert_dim(tensor: torch.Tensor, new_to_old: torch.Tensor) -> None:
    """Reorder dim 0 of a (possibly distributed) tensor in place: new row ``j`` = old row ``new_to_old[j]``.

    Args:
        tensor: Parameter or buffer of shape [experts, ...] with the expert dimension first. A ``DTensor``
            may use any placements (the full tensor is gathered, reordered and re-sharded with the same
            mesh and placements); its local shard is overwritten in place.
        new_to_old: Int64 tensor of shape [experts], a permutation of ``range(experts)``.

    Raises:
        RuntimeError: If the re-gathered tensor does not equal the expected reordering.
    """
    if isinstance(tensor, DTensor):
        full = tensor.full_tensor()
        expected = full.index_select(0, new_to_old.to(full.device))
        tensor.to_local().copy_(distribute_tensor(expected, tensor.device_mesh, tensor.placements).to_local())
        if not torch.equal(tensor.full_tensor(), expected):
            raise RuntimeError(f"expert permutation of a {tuple(full.shape)} DTensor did not round-trip")
    else:
        tensor.copy_(tensor.index_select(0, new_to_old.to(tensor.device)))


@torch.no_grad()
def apply_expert_placement(model: nn.Module, placement: dict) -> list[int]:
    """Relabel the experts of every MoE layer named in ``placement``, in place, on every rank.

    Must run on all ranks (it gathers distributed tensors) after sharding and before the optimizer
    creates state, so optimizer state is built from the permuted parameters.

    Args:
        model: Built and sharded model whose ``MoE`` modules live under ``...layers.<index>...``.
        placement: Output of :func:`compute_placement`.

    Returns:
        The layer ids that were permuted.

    Raises:
        ValueError: If the placement names a layer the model does not have or is not a permutation.
    """
    moe_by_layer = {}
    for name, module in model.named_modules():
        match = _LAYER_INDEX_RE.search(name + ".")
        if isinstance(module, MoE) and match:
            moe_by_layer[int(match.group(1))] = module
    applied = []
    for layer_key, spec in placement["layers"].items():
        layer_id = int(layer_key)
        if layer_id not in moe_by_layer:
            raise ValueError(f"placement names layer {layer_id}, which is not an MoE layer of this model")
        moe = moe_by_layer[layer_id]
        new_to_old = torch.tensor(spec["new_to_old"], dtype=torch.int64)
        if sorted(new_to_old.tolist()) != list(range(moe.gate.n_experts)):
            raise ValueError(f"placement for layer {layer_id} is not a permutation of {moe.gate.n_experts} experts")
        _permute_expert_dim(moe.gate.weight, new_to_old)
        if moe.gate.bias is not None:
            _permute_expert_dim(moe.gate.bias, new_to_old)
        if moe.gate.e_score_correction_bias is not None:
            _permute_expert_dim(moe.gate.e_score_correction_bias, new_to_old)
        for param in moe.experts.parameters(recurse=False):
            _permute_expert_dim(param, new_to_old)
        applied.append(layer_id)
    logger.info("expert placement applied to %d MoE layers: %s", len(applied), applied)
    return applied


def main() -> None:
    """Compute a placement from a derisk-v1 routing.npz and write it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    placement = compute_placement(args.routing, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(placement, indent=1))
    layers = placement["layers"].values()
    logger.info(
        "mean dispatch imbalance over layers: baseline %.3f -> predicted %.3f (max %.3f -> %.3f)",
        np.mean([v["baseline_imbalance_mean"] for v in layers]),
        np.mean([v["predicted_imbalance_mean"] for v in layers]),
        max(v["baseline_imbalance_max"] for v in layers),
        max(v["predicted_imbalance_max"] for v in layers),
    )


if __name__ == "__main__":
    main()
