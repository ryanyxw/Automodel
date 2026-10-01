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

"""NVTX ranges that show, per GPU, how long each MoE layer computes its experts and how long it waits for the others.

Every routed-expert block (``GroupedExpertsDeepEP``) runs three phases, in forward, in activation-checkpoint
recompute, and in backward:

* ``dispatch``: DeepEP all-to-all that sends each token to the GPUs that host its experts;
* ``experts``: the grouped GEMMs over the tokens this GPU received;
* ``combine``: DeepEP all-to-all that returns the expert outputs. Its kernels spin until every GPU has sent
  its results, so on a GPU with little expert work this range is mostly waiting for the busiest GPU.

Each phase gets an NVTX range in the ``MoE`` domain (its own row in Nsight Systems), named for example::

    L12 experts fwd | 212,874 tok | 2.17x avg
    L12 combine bwd | 0.61x avg

``tok`` is the number of token-to-expert assignments this GPU received in that layer, and ``x avg`` divides it
by the average over the expert-parallel GPUs. The average needs no communication: every GPU sends
``tokens * top_k`` assignments and the GPUs receive what they send in total, so with equal token counts per
GPU (true with fixed-length padding) the average received equals the local ``tokens * top_k``. The counts come
from DeepEP's dispatch, which already returns them to the host, so the labels add no GPU synchronization.
``experts`` ranges are colored by that ratio (green balanced, red hot), ``dispatch`` blue, ``combine`` purple;
the whole MoE layer (router and shared expert included) is a grey range around them.

Backward ranges are opened and closed by identity autograd functions placed on the hidden-state path around
each phase, so they bracket exactly the backward of that phase. Recompute ranges are labelled ``recompute``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import nvtx
import torch
from torch import nn

from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
from nemo_automodel.components.moe.layers import MoE

DOMAIN = "MoE"
_DISPATCH_COLOR = 0xFF1F77B4
_COMBINE_COLOR = 0xFF9467BD
_LAYER_COLOR = 0xFFBBBBBB
# (upper bound of received / average, color) for the experts ranges, first match wins.
_EXPERT_COLORS = ((1.25, 0xFF2CA02C), (1.75, 0xFFE8C020), (2.5, 0xFFFF7F0E), (float("inf"), 0xFFD62728))
_LAYER_INDEX = re.compile(r"layers\.(\d+)")


def expert_color(ratio: float) -> int:
    """Return the ARGB color of an ``experts`` range whose GPU received ``ratio`` times the average load."""
    return next(color for bound, color in _EXPERT_COLORS if ratio < bound)


def _mode() -> str:
    """Return ``"recompute"`` when running inside the autograd engine (activation checkpointing), else ``"fwd"``."""
    return "recompute" if torch._C._current_graph_task_id() != -1 else "fwd"


class _BackwardRange(torch.autograd.Function):
    """Identity in forward; pushes (``message`` given) or pops an NVTX range when the gradient passes through."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, message: str | None, color: int) -> torch.Tensor:
        """Return a view of ``x`` and remember what to do in backward.

        Args:
            ctx: Autograd context.
            x: Tensor of any shape; returned as a view (same shape, dtype, device, storage).
            message: Name of the range to push when the gradient arrives, or ``None`` to pop.
            color: ARGB color of the pushed range.

        Returns:
            ``x.view_as(x)``, a new autograd node that aliases ``x``.
        """
        ctx.message = message
        ctx.color = color
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Push or pop the NVTX range and pass the gradient through unchanged.

        Args:
            ctx: Autograd context from :meth:`forward`.
            grad: Gradient of the output, same shape as ``x`` in forward.

        Returns:
            ``(grad, None, None)``: the gradient for ``x``, none for ``message`` and ``color``.
        """
        if ctx.message is None:
            nvtx.pop_range(domain=DOMAIN)
        else:
            nvtx.push_range(ctx.message, color=ctx.color, domain=DOMAIN)
        return grad, None, None


def _mark(x: torch.Tensor, message: str | None, color: int = _LAYER_COLOR) -> torch.Tensor:
    """Insert a :class:`_BackwardRange` on ``x`` when autograd records the graph; otherwise return ``x``.

    Args:
        x: Tensor of any shape on the hidden-state path.
        message: Range to push when the gradient reaches this point in backward, or ``None`` to pop one.
        color: ARGB color of the pushed range.

    Returns:
        ``x`` or an aliasing view of it, same shape.
    """
    if torch.is_grad_enabled() and x.requires_grad:
        return _BackwardRange.apply(x, message, color)
    return x


class _LayerRanges:
    """Forward / recompute NVTX ranges of one MoE layer, closed even when the layer's forward is aborted.

    PyTorch's non-reentrant activation checkpointing (used for these blocks) stops a recompute early by raising
    an exception inside the block once every tensor backward needs has been recomputed. Ranges opened in that
    forward would then never be popped, and Nsight would pair every later pop with the wrong push. The layer's
    outer range therefore closes, in a ``finally``, every phase range still open inside it.
    """

    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.open = 0  # phase ranges pushed inside the current layer range and not popped yet
        # Filled by the forward dispatch, read by the matching combine: (label suffix, experts-bwd name, color).
        self.last: tuple[str, str, int] = ("", "", _LAYER_COLOR)

    def push(self, message: str, color: int) -> None:
        nvtx.push_range(message, color=color, domain=DOMAIN)
        self.open += 1

    def pop(self) -> None:
        if self.open > 0:
            nvtx.pop_range(domain=DOMAIN)
            self.open -= 1

    def pop_all(self) -> None:
        while self.open > 0:
            self.pop()


def _annotate_experts(ranges: _LayerRanges, experts: GroupedExpertsDeepEP) -> None:
    """Wrap one layer's token dispatcher so its dispatch / experts / combine phases emit NVTX ranges.

    The dispatcher is per module (``GroupedExpertsDeepEP.init_token_dispatcher``), so wrapping its bound
    methods on the instance labels exactly this layer.
    """
    layer = ranges.layer
    dispatcher = experts.token_dispatcher
    permute: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = dispatcher.token_permutation2
    unpermute: Callable[[torch.Tensor], torch.Tensor] = dispatcher.token_unpermutation

    def token_permutation2(
        hidden_states: torch.Tensor, num_local_tokens: int, token_probs: torch.Tensor, token_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch like the original, inside NVTX ranges; then open the ``experts`` range.

        Args:
            hidden_states: Tensor of shape [tokens, hidden], this GPU's tokens.
            num_local_tokens: ``tokens``.
            token_probs: Tensor of shape [tokens, top_k], routing weights.
            token_indices: Tensor of shape [tokens, top_k], global expert ids (-1 = not routed).

        Returns:
            The original ``(permuted_hidden_states, tokens_per_expert, permuted_probs)``:
            ``permuted_hidden_states`` of shape [received assignments, hidden] grouped by local expert,
            ``tokens_per_expert`` a CPU int tensor of shape [local experts], ``permuted_probs`` of shape
            [received assignments].
        """
        mode = _mode()
        hidden_states = _mark(hidden_states, None)  # backward: closes "dispatch bwd"
        ranges.push(f"L{layer:02d} dispatch {mode}", _DISPATCH_COLOR)
        try:
            permuted, tokens_per_expert, permuted_probs = permute(
                hidden_states, num_local_tokens, token_probs, token_indices
            )
        finally:
            ranges.pop()
        if tokens_per_expert.is_cuda:
            raise RuntimeError("expected DeepEP's tokens_per_expert on the host; reading it would sync the GPU")
        received = int(tokens_per_expert.sum())
        ratio = received / max(num_local_tokens * token_indices.shape[1], 1)
        color = expert_color(ratio)
        load = f"{ratio:.2f}x avg"
        ranges.last = (load, f"L{layer:02d} experts bwd | {received:,} tok | {load}", color)
        permuted = _mark(permuted, f"L{layer:02d} dispatch bwd | {load}", _DISPATCH_COLOR)
        permuted = _mark(permuted, None)  # backward: closes "experts bwd"
        ranges.push(f"L{layer:02d} experts {mode} | {received:,} tok | {load}", color)  # closed by the combine
        return permuted, tokens_per_expert, permuted_probs

    def token_unpermutation(hidden_states: torch.Tensor) -> torch.Tensor:
        """Close the ``experts`` range, then combine like the original inside NVTX ranges.

        Args:
            hidden_states: Tensor of shape [received assignments, hidden], expert outputs grouped by local expert.

        Returns:
            Tensor of shape [tokens, hidden]: the combined expert output for this GPU's tokens.
        """
        mode = _mode()
        load, experts_bwd, color = ranges.last
        ranges.pop()  # closes "experts fwd/recompute"
        hidden_states = _mark(hidden_states, experts_bwd, color)
        hidden_states = _mark(hidden_states, None)  # backward: closes "combine bwd"
        ranges.push(f"L{layer:02d} combine {mode} | {load}", _COMBINE_COLOR)
        try:
            out = unpermute(hidden_states)
        finally:
            ranges.pop()
        return _mark(out, f"L{layer:02d} combine bwd | {load}", _COMBINE_COLOR)

    dispatcher.token_permutation2 = token_permutation2
    dispatcher.token_unpermutation = token_unpermutation


def _annotate_layer(ranges: _LayerRanges, moe: nn.Module) -> None:
    """Wrap ``moe.forward`` in a grey ``L<layer> MoE fwd|recompute`` range that also closes any open phase range."""
    forward = moe.forward

    def annotated_forward(*args: object, **kwargs: object) -> object:
        nvtx.push_range(f"L{ranges.layer:02d} MoE {_mode()}", color=_LAYER_COLOR, domain=DOMAIN)
        try:
            return forward(*args, **kwargs)
        finally:
            ranges.pop_all()
            nvtx.pop_range(domain=DOMAIN)

    moe.forward = annotated_forward


def install_moe_nvtx(model: nn.Module) -> list[int]:
    """Annotate every DeepEP MoE layer of ``model`` with the NVTX ranges described in this module.

    Call once, after the model is sharded (the token dispatchers exist) and before training.

    Args:
        model: Model whose MoE layers live under ``...layers.<index>...`` module names.

    Returns:
        The annotated layer indices, in model order.

    Raises:
        ValueError: If ``model`` has no ``MoE`` layer with ``GroupedExpertsDeepEP`` experts.
    """
    layers = []
    for name, module in model.named_modules():
        if not isinstance(module, MoE) or not isinstance(module.experts, GroupedExpertsDeepEP):
            continue
        match = _LAYER_INDEX.search(name)
        if match is None:
            raise ValueError(f"cannot find a layer index in MoE module name {name!r}")
        layer = int(match.group(1))
        ranges = _LayerRanges(layer)
        _annotate_layer(ranges, module)
        _annotate_experts(ranges, module.experts)
        layers.append(layer)
    if not layers:
        raise ValueError("no MoE layer with DeepEP experts found")
    return layers
