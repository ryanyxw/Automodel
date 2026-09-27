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

"""Router hooks for the main-v1 padding study.

:class:`RoutingProbe` hooks the router (:class:`~nemo_automodel.components.moe.layers.Gate`) of every
MoE layer and counts, for every micro-batch, how many top-k routing assignments land on each expert,
split by token category. Category 0 is real (non-padding) tokens; categories 1.. are padding tokens
bucketed by their distance from the last real token of their sequence. At each optimizer step the
counts are summed over the data-parallel group, so they describe the global micro-batch that the
expert dispatcher saw.

Nemotron V3 does not pass its padding mask to the MoE layers, so these router decisions are also the
experts that padding tokens are actually dispatched to and computed on.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.hooks import RemovableHandle

from nemo_automodel.components.moe.layers import Gate

logger = logging.getLogger(__name__)

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class ProbeConfig:
    """Settings for the main-v1 probes (maps to the YAML ``probe:`` block).

    Attributes:
        output_dir: Directory for probe outputs (routing tensors, per-step records, run metadata).
        pad_distance_edges: Increasing bucket edges for a pad token's distance from the last real token
            of its sequence; the first pad has distance 0 and bucket k covers [edges[k-1], edges[k]).
        summary_warmup_steps: Leading steps excluded from the run summary averages.
        nsys_start_step: First 0-based step of the nsys capture window, or -1 to disable capture.
        nsys_end_step: Last 0-based step of the nsys capture window (inclusive).
        nsys_ranks: Global ranks that call ``cudaProfilerStart``/``cudaProfilerStop``.
    """

    output_dir: str
    pad_distance_edges: tuple[int, ...] = (1, 8, 64, 512, 4096)
    summary_warmup_steps: int = 2
    nsys_start_step: int = -1
    nsys_end_step: int = -1
    nsys_ranks: tuple[int, ...] = (0,)

    def build(self, *, model: nn.Module, ep_size: int, device: torch.device) -> RoutingProbe:
        """Attach a :class:`RoutingProbe` to ``model``.

        Args:
            model: The (sharded) model whose MoE routers are hooked.
            ep_size: Expert-parallel size; expert ``e`` lives on EP rank ``e // (n_experts // ep_size)``.
            device: Device of the router outputs; counting buffers live here.

        Returns:
            The attached probe. Call :meth:`RoutingProbe.close` to remove its hooks.
        """
        return RoutingProbe(model, pad_distance_edges=self.pad_distance_edges, ep_size=ep_size, device=device)


def pad_category_names(pad_distance_edges: tuple[int, ...]) -> list[str]:
    """Return display names for the token categories counted by :class:`RoutingProbe`.

    Args:
        pad_distance_edges: Increasing pad-distance bucket edges, as in :class:`ProbeConfig`.

    Returns:
        ``["real", <one name per pad-distance bucket>]``, e.g. ``pad[0]``, ``pad[1,8)``, ``pad[4096,inf)``.
    """
    names = ["real"]
    lower = 0
    for upper in pad_distance_edges:
        names.append(f"pad[{lower}]" if upper == lower + 1 else f"pad[{lower},{upper})")
        lower = upper
    names.append(f"pad[{lower},inf)")
    return names


def token_categories(attention_mask: torch.Tensor, pad_distance_edges: torch.Tensor) -> torch.Tensor:
    """Assign every token of a right-padded micro-batch to a routing category.

    Args:
        attention_mask: Tensor of shape [batch, sequence] on the CPU; 1 marks real tokens and 0 marks
            padding. Every row must be right-padded (a run of ones followed by a run of zeros).
        pad_distance_edges: Tensor of shape [buckets - 1] with increasing int64 bucket edges for a pad
            token's distance from the last real token of its row (the first pad has distance 0).

    Returns:
        Int64 tensor of shape [batch * sequence], the row-major flattening of [batch, sequence] that
        the MoE layers use as their token order. Real tokens get category 0; a pad token at distance
        ``d`` gets category ``k + 1`` where ``k`` is the bucket with ``edges[k-1] <= d < edges[k]``.

    Raises:
        ValueError: If a row of ``attention_mask`` is not right-padded.
    """
    mask = attention_mask.to(torch.bool)
    positions = torch.arange(mask.shape[1]).unsqueeze(0)
    real_lengths = mask.sum(dim=1, keepdim=True)
    if not torch.equal(mask, positions < real_lengths):
        raise ValueError("attention_mask rows must be right-padded (ones followed by zeros)")
    pad_bucket = torch.bucketize(positions - real_lengths, pad_distance_edges, right=True) + 1
    return torch.where(mask, torch.zeros_like(pad_bucket), pad_bucket).reshape(-1)


def summarize_routing(counts: np.ndarray, *, ep_size: int, top_k: int, layer_ids: list[int]) -> dict[str, float]:
    """Summarize one optimizer step of global router counts into scalar metrics.

    Args:
        counts: Array of shape [microbatches, layers, categories, experts] holding top-k assignment
            counts summed over all data-parallel ranks; category 0 is real tokens and categories 1..
            are pad-distance buckets.
        ep_size: Expert-parallel size; expert ``e`` lives on EP rank ``e // (experts // ep_size)``.
        top_k: Number of experts each token is routed to.
        layer_ids: Model layer index of each probed MoE layer, in ``counts`` layer order.

    Returns:
        Scalar metrics keyed for wandb. ``ep_imbalance`` is the max over EP ranks of the assignments
        routed to that rank's experts divided by the mean over ranks (1.0 is perfectly balanced,
        ``ep_size`` means one rank gets everything), taken per micro-batch because each micro-batch is
        one dispatch. ``top{k}_share`` is the fraction of a layer's assignments captured by its ``k``
        most-used experts (``k / experts`` when routing is uniform).
    """
    n_micro, n_layers, _, n_experts = counts.shape
    real = counts[:, :, 0, :].astype(np.float64)
    pad = counts[:, :, 1:, :].sum(axis=2).astype(np.float64)

    def top_share(per_expert: np.ndarray) -> np.ndarray:
        top = np.sort(per_expert, axis=-1)[..., ::-1][..., :top_k].sum(axis=-1)
        return top / np.maximum(per_expert.sum(axis=-1), 1.0)

    def normalized_entropy(per_expert: np.ndarray) -> np.ndarray:
        prob = per_expert / np.maximum(per_expert.sum(axis=-1, keepdims=True), 1.0)
        plogp = np.where(prob > 0, prob * np.log(np.where(prob > 0, prob, 1.0)), 0.0)
        return -plogp.sum(axis=-1) / math.log(n_experts)

    def ep_imbalance(per_expert: np.ndarray) -> np.ndarray:
        loads = per_expert.reshape(n_micro, n_layers, ep_size, n_experts // ep_size).sum(axis=-1)
        return loads.max(axis=-1) / np.maximum(loads.mean(axis=-1), 1e-9)

    step_real, step_pad = real.sum(axis=0), pad.sum(axis=0)
    imbalance_all = ep_imbalance(real + pad)
    pad_share = top_share(step_pad)
    metrics = {
        "routing/pad_assignment_fraction": pad.sum() / max((real + pad).sum(), 1.0),
        f"routing/pad_top{top_k}_share_mean": pad_share.mean(),
        f"routing/real_top{top_k}_share_mean": top_share(step_real).mean(),
        "routing/pad_entropy_norm_mean": normalized_entropy(step_pad).mean(),
        "routing/real_entropy_norm_mean": normalized_entropy(step_real).mean(),
        "routing/ep_imbalance_all_max": imbalance_all.max(),
        "routing/ep_imbalance_all_mean": imbalance_all.mean(),
        "routing/ep_imbalance_real_mean": ep_imbalance(real).mean(),
        "routing/ep_imbalance_pad_mean": ep_imbalance(pad).mean(),
    }
    for slot, layer_id in enumerate(layer_ids):
        metrics[f"routing_layer/L{layer_id:02d}_ep_imbalance_all"] = imbalance_all[:, slot].max()
        metrics[f"routing_layer/L{layer_id:02d}_pad_top{top_k}_share"] = pad_share[slot]
    return {key: float(value) for key, value in metrics.items()}


class RoutingProbe:
    """Counts router top-k assignments per micro-batch, MoE layer, token category, and expert.

    Hooks registered on ``model``:

    * forward pre/post hooks on the model itself bracket the real forward pass, so router calls that
      activation-checkpoint recomputation replays during backward are not counted twice;
    * a forward hook on every :class:`Gate` accumulates its ``indices`` and ``weights`` outputs into
      device buffers with ``index_add_`` (no host synchronization in the forward pass).

    Call :meth:`begin_microbatch` before every training micro-batch forward and :meth:`end_step` on
    every rank after every optimizer step (it runs collectives).
    """

    def __init__(
        self, model: nn.Module, *, pad_distance_edges: tuple[int, ...], ep_size: int, device: torch.device
    ) -> None:
        """Hook every MoE router of ``model``.

        Args:
            model: Model containing :class:`Gate` modules under ``...layers.<index>...`` names.
            pad_distance_edges: Increasing pad-distance bucket edges, as in :class:`ProbeConfig`.
            ep_size: Expert-parallel size; must divide the number of routed experts.
            device: Device of the router outputs; counting buffers live here.

        Raises:
            ValueError: If ``model`` has no :class:`Gate` modules or ``ep_size`` does not divide the
                number of routed experts.
        """
        gates = [(name, module) for name, module in model.named_modules() if isinstance(module, Gate)]
        if not gates:
            raise ValueError(f"{type(model).__name__} has no MoE Gate modules to probe")
        self.layer_names = [name for name, _ in gates]
        self.layer_ids = [int(_LAYER_INDEX_RE.search(name).group(1)) for name in self.layer_names]
        self.n_experts = int(gates[0][1].n_experts)
        self.top_k = int(gates[0][1].topk)
        if self.n_experts % ep_size != 0:
            raise ValueError(f"ep_size={ep_size} must divide n_experts={self.n_experts}")
        self.ep_size = ep_size
        self.pad_distance_edges = tuple(int(edge) for edge in pad_distance_edges)
        self.category_names = pad_category_names(self.pad_distance_edges)
        self.device = device
        self._edges = torch.tensor(self.pad_distance_edges, dtype=torch.int64)
        self._in_model_forward = False
        self._category: torch.Tensor | None = None
        # Per micro-batch of the current step: [layers, categories * experts] device buffers.
        self._counts: list[torch.Tensor] = []
        self._weight_sums: list[torch.Tensor] = []
        # Per micro-batch of the current step on this rank: (batch, sequence, real tokens).
        self._shapes: list[tuple[int, int, int]] = []
        self._gate_calls = [0] * len(gates)
        self.history_steps: list[int] = []
        self.history_counts: list[np.ndarray] = []
        self.history_weight_sums: list[np.ndarray] = []
        self.history_shapes: list[np.ndarray] = []
        self._handles: list[RemovableHandle] = [
            model.register_forward_pre_hook(self._on_model_forward_start),
            model.register_forward_hook(self._on_model_forward_end),
        ]
        for slot, (_, gate) in enumerate(gates):
            self._handles.append(gate.register_forward_hook(partial(self._on_gate_forward, slot)))
        logger.info(
            "RoutingProbe hooked %d MoE routers (layers %s), %d experts, top-%d, EP=%d",
            len(gates),
            self.layer_ids,
            self.n_experts,
            self.top_k,
            ep_size,
        )

    def begin_microbatch(self, index: int, attention_mask: torch.Tensor) -> None:
        """Prepare the counting buffers for one training micro-batch.

        Args:
            index: 0-based micro-batch index within the optimizer step; must equal the number of
                micro-batches already begun in this step.
            attention_mask: Tensor of shape [batch, sequence] (any device); 1 marks real tokens and 0
                marks right padding.

        Raises:
            RuntimeError: If micro-batches arrive out of order.
        """
        if index != len(self._counts):
            raise RuntimeError(f"expected micro-batch {len(self._counts)} of the step, got {index}")
        mask = attention_mask.detach().cpu()
        self._category = token_categories(mask, self._edges).to(self.device, non_blocking=True)
        buffer_shape = (len(self.layer_ids), len(self.category_names) * self.n_experts)
        self._counts.append(torch.zeros(buffer_shape, dtype=torch.int64, device=self.device))
        self._weight_sums.append(torch.zeros(buffer_shape, dtype=torch.float64, device=self.device))
        self._shapes.append((mask.shape[0], mask.shape[1], int(mask.sum())))

    def end_step(self, step: int, group: dist.ProcessGroup | None) -> dict[str, float]:
        """Sum this step's counts over ``group``, archive them, and return summary metrics.

        Must be called on every rank of ``group`` after every optimizer step.

        Args:
            step: 0-based optimizer step index, stored alongside the archived tensors.
            group: Data-parallel process group spanning every rank that holds different tokens.

        Returns:
            Scalar routing metrics from :func:`summarize_routing`.

        Raises:
            RuntimeError: On every rank, if any rank saw a router fire other than exactly once per
                micro-batch forward (the recomputation filter failed, so the counts are invalid).
        """
        n_micro = len(self._counts)
        miscounted = torch.tensor([int(any(calls != n_micro for calls in self._gate_calls))], device=self.device)
        dist.all_reduce(miscounted, op=dist.ReduceOp.MAX, group=group)
        if miscounted.item():
            raise RuntimeError(f"router hooks fired {self._gate_calls} times for {n_micro} micro-batches on some rank")
        counts = torch.stack(self._counts)
        weight_sums = torch.stack(self._weight_sums)
        dist.all_reduce(counts, group=group)
        dist.all_reduce(weight_sums, group=group)
        local_shapes = torch.tensor(self._shapes, dtype=torch.int64, device=self.device)
        all_shapes = torch.empty(
            (dist.get_world_size(group), *local_shapes.shape), dtype=torch.int64, device=self.device
        )
        dist.all_gather_into_tensor(all_shapes, local_shapes, group=group)

        per_category = (n_micro, len(self.layer_ids), len(self.category_names), self.n_experts)
        counts_np = counts.view(per_category).cpu().numpy()
        self.history_steps.append(step)
        self.history_counts.append(counts_np)
        self.history_weight_sums.append(weight_sums.view(per_category).cpu().numpy())
        self.history_shapes.append(all_shapes.cpu().numpy())

        self._counts, self._weight_sums, self._shapes = [], [], []
        self._gate_calls = [0] * len(self._gate_calls)
        self._category = None
        return summarize_routing(counts_np, ep_size=self.ep_size, top_k=self.top_k, layer_ids=self.layer_ids)

    def save(self, path: Path) -> None:
        """Write the archived per-step tensors to a compressed ``.npz`` file.

        Arrays written: ``steps`` [steps]; ``counts`` [steps, microbatches, layers, categories, experts]
        int64 top-k assignment counts summed over ranks; ``routing_weight_sums`` (same shape, float64)
        sums of the router weights of those assignments; ``microbatch_shapes`` [steps, ranks,
        microbatches, 3] int64 rows of (batch, padded sequence, real tokens); ``layer_ids`` [layers];
        ``metadata`` a JSON string with category names, expert counts, and EP layout.

        Args:
            path: Destination file.
        """
        if not self.history_steps:
            logger.warning("RoutingProbe recorded no steps; skipping %s", path)
            return
        metadata = {
            "category_names": self.category_names,
            "pad_distance_edges": list(self.pad_distance_edges),
            "layer_names": self.layer_names,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "ep_size": self.ep_size,
            "experts_per_ep_rank": self.n_experts // self.ep_size,
        }
        np.savez_compressed(
            path,
            steps=np.asarray(self.history_steps, dtype=np.int64),
            counts=np.stack(self.history_counts),
            routing_weight_sums=np.stack(self.history_weight_sums),
            microbatch_shapes=np.stack(self.history_shapes),
            layer_ids=np.asarray(self.layer_ids, dtype=np.int64),
            metadata=np.asarray(json.dumps(metadata)),
        )

    def close(self) -> None:
        """Remove every hook this probe registered."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _on_model_forward_start(self, module: nn.Module, args: tuple[object, ...]) -> None:
        self._in_model_forward = True

    def _on_model_forward_end(self, module: nn.Module, args: tuple[object, ...], output: object) -> None:
        self._in_model_forward = False

    def _on_gate_forward(
        self,
        slot: int,
        module: nn.Module,
        args: tuple[object, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ) -> None:
        """Accumulate one router call into the current micro-batch buffers.

        Args:
            slot: Position of the hooked gate in ``self.layer_ids``.
            module: The hooked :class:`Gate`.
            args: Gate inputs ``(x, token_mask, cp_mesh)``, where ``x`` is a tensor of shape
                [tokens, hidden] and ``token_mask`` a bool tensor of shape [tokens].
            output: Gate outputs ``(weights, indices, aux_loss)``: ``weights`` (float) and ``indices``
                (int64 expert ids) are tensors of shape [tokens, top_k] in the micro-batch's row-major
                token order; ``aux_loss`` is a scalar tensor or ``None``.

        Raises:
            RuntimeError: If the router saw a different number of tokens than the micro-batch mask.
        """
        if not self._in_model_forward:
            return
        weights, indices = output[0], output[1]
        if self._category is None or indices.shape[0] != self._category.numel():
            expected = None if self._category is None else self._category.numel()
            raise RuntimeError(f"router saw {indices.shape[0]} tokens, micro-batch mask has {expected}")
        flat = (self._category.unsqueeze(1) * self.n_experts + indices).reshape(-1)
        self._counts[-1][slot].index_add_(0, flat, torch.ones_like(flat))
        self._weight_sums[-1][slot].index_add_(0, flat, weights.detach().reshape(-1).to(torch.float64))
        self._gate_calls[slot] += 1
