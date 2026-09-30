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

"""CPU checks for main-v1 (run: python experiments/main-v1/test_main_v1.py).

1. The padding methods rewrite only padding positions: real tokens, labels and (for hidden padding) masks
   are unchanged, and the draws are reproducible.
2. Relabeling the experts of a Nemotron-V3-style MoE layer (sigmoid router with score-correction bias,
   relu2 experts, shared expert) leaves its output unchanged, also with LoRA adapters on the experts
   (main-v1.1), and the placement search spreads experts that always receive the same tokens over different GPUs.
3. The router probe assigns token categories and computes the GPU load imbalance as documented.
4. On the real training data, the collate functions produce the shapes and pad token the report describes.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from expert_placement import apply_expert_placement, compute_layer_placement, dispatch_imbalance
from pad_fill import natural_pad_collater, random_pad_collater
from padding_probe import summarize_routing, token_categories
from torch import nn

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig
from nemo_automodel.components.datasets.utils import default_collater
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_PAD = 11


def _samples() -> list[dict[str, list[int]]]:
    """Two right-paddable samples of lengths 5 and 3 in ChatDataset's list format."""
    return [
        {
            "input_ids": [5000, 5001, 5002, 5003, 5004],
            "labels": [-100, -100, 5002, 5003, 5004],
            "attention_mask": [1, 1, 1, 1, 1],
            "___PAD_TOKEN_IDS___": {"input_ids": _PAD, "labels": -100, "attention_mask": 0},
        },
        {"input_ids": [6000, 6001, 6002], "labels": [-100, 6001, 6002], "attention_mask": [1, 1, 1]},
    ]


def test_random_pad_rewrites_only_padding() -> None:
    """Random pads: every padding id is redrawn within [low, high); everything else equals default_collater."""
    reference = default_collater(_samples(), pad_seq_len_divisible=8)
    out = random_pad_collater(_samples(), pad_seq_len_divisible=8, low=1000, high=2000, seed=7)
    padding = reference["attention_mask"] == 0
    assert out["input_ids"].shape == (2, 8)
    assert torch.equal(out["input_ids"][~padding], reference["input_ids"][~padding])
    assert bool(((out["input_ids"][padding] >= 1000) & (out["input_ids"][padding] < 2000)).all())
    for key in ("labels", "attention_mask", "padding_mask"):
        assert torch.equal(out[key], reference[key]), key
    again = random_pad_collater(_samples(), pad_seq_len_divisible=8, low=1000, high=2000, seed=7)
    assert torch.equal(out["input_ids"], again["input_ids"]), "same batch and seed must give the same pads"


def test_natural_pad_masked_and_visible() -> None:
    """Natural filler: contiguous stream windows in the padding; visibility only changes the masks."""
    stream = np.arange(100, 200, dtype=np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "filler.npy")
        np.save(path, stream)
        reference = default_collater(_samples(), pad_seq_len_divisible=8)
        hidden = natural_pad_collater(_samples(), filler_path=path, visible=False, pad_seq_len_divisible=8)
        shown = natural_pad_collater(_samples(), filler_path=path, visible=True, pad_seq_len_divisible=8)
    padding = reference["attention_mask"] == 0
    assert torch.equal(hidden["input_ids"], shown["input_ids"]), "visibility must not change the filler text"
    assert torch.equal(hidden["input_ids"][~padding], reference["input_ids"][~padding])
    for row in range(2):
        window = hidden["input_ids"][row][padding[row]]
        assert torch.equal(window, torch.arange(int(window[0]), int(window[0]) + window.numel())), "not contiguous"
    assert torch.equal(hidden["real_mask"], reference["attention_mask"])
    assert torch.equal(shown["real_mask"], reference["attention_mask"])
    assert torch.equal(hidden["attention_mask"], reference["attention_mask"])
    assert torch.equal(hidden["padding_mask"], reference["padding_mask"])
    assert bool(shown["attention_mask"].eq(1).all()) and not bool(shown["padding_mask"].any())
    assert torch.equal(hidden["labels"], reference["labels"]) and torch.equal(shown["labels"], reference["labels"])


class _Wrapper(nn.Module):
    """Puts an MoE under ``layers.<index>.mixer`` so the placement's layer ids resolve."""

    def __init__(self, moe: MoE, layer_id: int) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(layer_id): nn.ModuleDict({"mixer": moe})})


def _nemotron_style_moe() -> MoE:
    """A small MoE layer shaped like Nemotron V3's: sigmoid router with score-correction bias, relu2 experts."""
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
    return moe


def test_relabeling_preserves_output() -> None:
    """Relabeling experts, router rows and router bias together must not change the MoE output."""
    torch.manual_seed(0)
    moe = _nemotron_style_moe()
    x = torch.randn(2, 5, 32)
    expected = moe(x)
    new_to_old = torch.randperm(16).tolist()
    applied = apply_expert_placement(_Wrapper(moe, 7), {"layers": {"7": {"new_to_old": new_to_old}}})
    assert applied == [7]
    torch.testing.assert_close(moe(x), expected, rtol=1e-5, atol=1e-6)


def test_relabeling_preserves_output_with_lora() -> None:
    """With LoRA adapters on the experts (main-v1.1), relabeling must move them too and keep the output."""
    torch.manual_seed(0)
    moe = _nemotron_style_moe()
    moe.experts = patch_moe_module(moe.experts, dim=4, alpha=8)
    with torch.no_grad():  # B starts at zero; make the adapters matter so a missed permutation would show
        for name, param in moe.experts.named_parameters(recurse=False):
            if name.startswith("lora_"):
                param.copy_(torch.randn_like(param) * 0.3)
    x = torch.randn(2, 5, 32)
    expected = moe(x)
    apply_expert_placement(_Wrapper(moe, 7), {"layers": {"7": {"new_to_old": torch.randperm(16).tolist()}}})
    torch.testing.assert_close(moe(x), expected, rtol=1e-5, atol=1e-6)


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


def test_probe_categories_and_imbalance() -> None:
    """Categories follow the documented pad-distance buckets; imbalance is busiest GPU over the mean GPU."""
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    categories = token_categories(mask, torch.tensor([1, 8])).view(2, 10)
    # Row 0: pads at distances 0..7 -> bucket [0] then [1,8), and the last pad (distance 7) is still [1,8).
    assert categories[0].tolist() == [0, 0, 1, 2, 2, 2, 2, 2, 2, 2]
    assert categories[1].tolist() == [0] * 9 + [1]
    # One micro-batch, one layer, 2 categories, 4 experts on 2 GPUs: GPU 0 gets 6 + 2, GPU 1 gets 0 + 0.
    counts = np.zeros((1, 1, 2, 4))
    counts[0, 0, 0] = [3, 3, 0, 0]
    counts[0, 0, 1] = [2, 0, 0, 0]
    metrics = summarize_routing(counts, ep_size=2, top_k=1, layer_ids=[5])
    assert metrics["routing/ep_imbalance_all_mean"] == 2.0  # busiest 8 / mean 4
    assert metrics["routing/pad_assignment_fraction"] == 0.25


def test_real_data_batches() -> None:
    """The first training micro-batch: default pads are <|im_end|>, fixed padding gives [2, 8192]."""
    cfg = yaml.safe_load((_HERE / "configs" / "fixed_default.yaml").read_text())
    fields = {key: value for key, value in cfg["dataset"].items() if key != "_target_"}
    tokenizer = NeMoAutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_name_or_path"], trust_remote_code=True)
    dataset = ChatDatasetConfig(**fields).build(tokenizer=tokenizer)
    pair = [dataset[0], dataset[8]]  # rank 0's first micro-batch: the sampler gives rank r samples r, r + 8, ...
    lengths = [len(sample["input_ids"]) for sample in pair]
    fixed = default_collater([dict(sample) for sample in pair], pad_seq_len_divisible=8192)
    batchmax = default_collater([dict(sample) for sample in pair])
    assert fixed["input_ids"].shape == (2, 8192) and batchmax["input_ids"].shape == (2, max(lengths))
    padding = fixed["attention_mask"] == 0
    pad_ids = set(fixed["input_ids"][padding].tolist())
    assert pad_ids == {tokenizer.convert_tokens_to_ids("<|im_end|>")}, pad_ids
    assert bool((fixed["labels"][padding] == -100).all())
    logger.info("first micro-batch: lengths %s, pad token %s", lengths, tokenizer.convert_ids_to_tokens(list(pad_ids)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_random_pad_rewrites_only_padding()
    test_natural_pad_masked_and_visible()
    test_relabeling_preserves_output()
    test_relabeling_preserves_output_with_lora()
    test_placement_spreads_hot_experts()
    test_probe_categories_and_imbalance()
    test_real_data_batches()
    logger.info("main-v1 checks passed")
