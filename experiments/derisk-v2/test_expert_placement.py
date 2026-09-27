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

"""CPU checks for derisk-v2.2's expert placement (run: python experiments/derisk-v2/test_expert_placement.py).

1. Relabeling the experts of a Nemotron-V3-style MoE layer (sigmoid router with score-correction bias,
   relu2 experts, shared expert) leaves its output unchanged.
2. ``compute_layer_placement`` spreads experts that always receive the same tokens over different ranks
   and never does worse than the contiguous layout.
"""

from __future__ import annotations

import numpy as np
import torch
from expert_placement import apply_expert_placement, compute_layer_placement, dispatch_imbalance
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE


class _Wrapper(nn.Module):
    """Puts an MoE under ``layers.<index>.mixer`` so the placement's layer ids resolve."""

    def __init__(self, moe: MoE, layer_id: int) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(layer_id): nn.ModuleDict({"mixer": moe})})


def test_relabeling_preserves_output() -> None:
    """Relabeling experts, router rows and router bias together must not change the MoE output."""
    torch.manual_seed(0)
    config = MoEConfig(
        dim=32,
        inter_dim=16,
        moe_inter_dim=16,
        n_routed_experts=16,
        n_shared_experts=1,
        n_activated_experts=4,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=2.5,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="relu2",
        dtype=torch.float32,
        shared_expert_gate=False,
        shared_expert_inter_dim=24,
        shared_expert_activation="relu2",
        force_e_score_correction_bias=True,
    )
    backend = BackendConfig(linear="torch", attn="sdpa", experts="torch", dispatcher="torch", rope_fusion=False)
    moe = MoE(config, backend)
    with torch.no_grad():
        moe.init_weights(torch.device("cpu"), init_std=0.2)
        moe.gate.e_score_correction_bias.copy_(torch.randn(16) * 0.1)
    x = torch.randn(2, 5, 32)
    expected = moe(x)
    new_to_old = torch.randperm(16).tolist()
    applied = apply_expert_placement(_Wrapper(moe, 7), {"layers": {"7": {"new_to_old": new_to_old}}})
    assert applied == [7]
    actual = moe(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_placement_spreads_hot_experts() -> None:
    """Always-hot experts must land on different ranks, with equal experts per rank and lower imbalance."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(10.0, size=(40, 16)).astype(float)
    counts[:, [0, 1, 2, 3]] += 1000.0  # four experts that every dispatch hits hard, all on rank 0 of 4
    rank_of_expert, value = compute_layer_placement(counts, 4, iterations=2000, rng=rng)
    assert sorted(set(rank_of_expert[[0, 1, 2, 3]].tolist())) == [0, 1, 2, 3], "hot experts share a rank"
    assert all((rank_of_expert == r).sum() == 4 for r in range(4)), "unequal experts per rank"
    baseline = dispatch_imbalance(counts.reshape(40, 4, 4).sum(axis=2)).mean()
    assert value < baseline and value < 1.05


if __name__ == "__main__":
    test_relabeling_preserves_output()
    test_placement_spreads_hot_experts()
    print("expert placement checks passed")
