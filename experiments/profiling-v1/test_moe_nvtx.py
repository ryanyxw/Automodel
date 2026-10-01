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

"""CPU checks of moe_nvtx: ranges nest correctly and bracket the right phase in forward and backward.

Run: python -m pytest experiments/profiling-v1/test_moe_nvtx.py
"""

from __future__ import annotations

from types import SimpleNamespace

import moe_nvtx
import pytest
import torch
import torch.utils.checkpoint
from torch import nn

EVENTS: list[str] = []


class _Logged(torch.autograd.Function):
    """Identity that logs ``<name>`` in forward and ``<name> grad`` in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str) -> torch.Tensor:
        """Log and return ``x * 1`` (x: tensor of any shape)."""
        ctx.name = name
        EVENTS.append(name)
        return x * 1

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Log and pass ``grad`` (same shape as ``x``) through."""
        EVENTS.append(f"{ctx.name} grad")
        return grad, None


class _FakeDispatcher:
    """Stands in for MoEFlexTokenDispatcher: 4 local experts, every token goes to this GPU twice (top-2)."""

    def token_permutation2(
        self, hidden_states: torch.Tensor, num_local_tokens: int, token_probs: torch.Tensor, token_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``hidden_states`` [tokens, hidden] repeated top_k times, CPU counts [4], probs [tokens * top_k]."""
        permuted = _Logged.apply(hidden_states.repeat(token_indices.shape[1], 1), "dispatch")
        tokens_per_expert = torch.tensor([num_local_tokens, num_local_tokens, 0, 0])
        return permuted, tokens_per_expert, token_probs.flatten()

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Sum the top_k copies of [tokens * top_k, hidden] back to [tokens, hidden]."""
        return _Logged.apply(hidden_states.view(2, -1, hidden_states.shape[-1]).sum(0), "combine")


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every NVTX push / pop in the ``MoE`` domain into ``EVENTS``."""
    EVENTS.clear()

    def push_range(message: str, color: int, domain: str) -> None:
        assert domain == moe_nvtx.DOMAIN
        EVENTS.append(f"push {message}")

    def pop_range(domain: str) -> None:
        assert domain == moe_nvtx.DOMAIN
        EVENTS.append("pop")

    monkeypatch.setattr(moe_nvtx.nvtx, "push_range", push_range)
    monkeypatch.setattr(moe_nvtx.nvtx, "pop_range", pop_range)
    return EVENTS


def _run_layer(x: torch.Tensor) -> torch.Tensor:
    """Run one annotated fake layer 7 on x [tokens, hidden] with top-2 routing; return [tokens, hidden]."""
    experts = SimpleNamespace(token_dispatcher=_FakeDispatcher())
    moe_nvtx._annotate_experts(moe_nvtx._LayerRanges(7), experts)
    dispatcher = experts.token_dispatcher
    tokens = x.shape[0]
    probs = torch.full((tokens, 2), 0.5)
    indices = torch.zeros((tokens, 2), dtype=torch.long)
    permuted, _, _ = dispatcher.token_permutation2(x, tokens, probs, indices)
    return dispatcher.token_unpermutation(_Logged.apply(permuted * 2, "experts"))


def test_forward_and_backward_ranges_bracket_each_phase(recorded: list[str]) -> None:
    """Forward ranges wrap each phase; backward ranges wrap exactly that phase's gradient, in reverse order."""
    x = torch.randn(3, 5, requires_grad=True)
    out = _run_layer(x)
    # 3 tokens * top-2 = 6 assignments sent; this GPU received 3 + 3 = 6 -> exactly the average.
    assert recorded == [
        "push L07 dispatch fwd",
        "dispatch",
        "pop",
        "push L07 experts fwd | 6 tok | 1.00x avg",
        "experts",
        "pop",
        "push L07 combine fwd | 1.00x avg",
        "combine",
        "pop",
    ]
    recorded.clear()
    out.sum().backward()
    assert recorded == [
        "push L07 combine bwd | 1.00x avg",
        "combine grad",
        "pop",
        "push L07 experts bwd | 6 tok | 1.00x avg",
        "experts grad",
        "pop",
        "push L07 dispatch bwd | 1.00x avg",
        "dispatch grad",
        "pop",
    ]


def test_gradients_are_unchanged(recorded: list[str]) -> None:
    """The backward markers pass gradients through untouched."""
    x = torch.randn(3, 5, requires_grad=True)
    _run_layer(x).sum().backward()
    # out = sum over 2 copies of (x * 2) = 4 x, so d(sum out)/dx = 4.
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.0))


def test_no_backward_ranges_without_grad(recorded: list[str]) -> None:
    """Under no_grad only the three forward ranges are emitted, and all are closed."""
    with torch.no_grad():
        _run_layer(torch.randn(3, 5))
    assert sum(event.startswith("push") for event in recorded) == 3
    assert recorded.count("pop") == 3


class _FakeMoE(nn.Module):
    """Stands in for an MoE layer: fake dispatch, a linear "expert", fake combine, then another linear."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = SimpleNamespace(token_dispatcher=_FakeDispatcher())
        self.expert = nn.Linear(5, 5)
        self.after = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: tensor of shape [tokens, 5]; returns [tokens, 5]."""
        dispatcher = self.experts.token_dispatcher
        tokens = x.shape[0]
        permuted, _, _ = dispatcher.token_permutation2(
            x, tokens, torch.full((tokens, 2), 0.5), torch.zeros((tokens, 2), dtype=torch.long)
        )
        # With early stop, the recompute raises inside ``after`` (the last op that saves a tensor), so the
        # layer's forward never returns normally.
        return self.after(dispatcher.token_unpermutation(self.expert(permuted)))


def test_ranges_stay_balanced_when_checkpoint_recompute_stops_early(recorded: list[str]) -> None:
    """An early-stopped checkpoint recompute (an exception inside the layer) still closes every range it opened."""
    moe = _FakeMoE()
    ranges = moe_nvtx._LayerRanges(7)
    moe_nvtx._annotate_layer(ranges, moe)
    moe_nvtx._annotate_experts(ranges, moe.experts)
    x = torch.randn(3, 5, requires_grad=True)
    with torch.utils.checkpoint.set_checkpoint_early_stop(True):
        out = torch.utils.checkpoint.checkpoint(moe, x, use_reentrant=False)
        out.sum().backward()
    pushes = [event for event in recorded if event.startswith("push")]
    assert "push L07 MoE recompute" in pushes
    assert len(pushes) == recorded.count("pop")
    depth = 0
    for event in recorded:
        depth += 1 if event.startswith("push") else -1 if event == "pop" else 0
        assert depth >= 0


@pytest.mark.parametrize(
    ("ratio", "color"), [(1.0, 0xFF2CA02C), (1.5, 0xFFE8C020), (2.0, 0xFFFF7F0E), (3.9, 0xFFD62728)]
)
def test_expert_color_scale(ratio: float, color: int) -> None:
    """Expert ranges go green -> yellow -> orange -> red as the load ratio grows."""
    assert moe_nvtx.expert_color(ratio) == color
