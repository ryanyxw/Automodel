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

"""derisk-v2.1 collate function: pad with random ordinary tokens instead of one repeated pad token.

In derisk-v1 every padding position held ``<|im_end|>`` with attention mask 0. Attention and Mamba mask
padding, so every pad token reached every MoE layer with the same hidden state and the router sent all
of them to the same 6 experts. Drawing a random token id per padding position gives each pad token its
own embedding, so the router spreads them over experts. Masks and labels are unchanged, so the loss on
real tokens is unchanged.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch

from nemo_automodel.components.datasets.utils import default_collater


def random_pad_collater(
    batch: list[dict[str, list[int] | torch.Tensor]],
    *,
    pad_seq_len_divisible: int | None = None,
    low: int = 1000,
    high: int = 131072,
    seed: int = 1234,
) -> dict[str, torch.Tensor]:
    """Collate like ``default_collater``, then overwrite the padding positions of ``input_ids``.

    Args:
        batch: Per-sample dictionaries as produced by ``ChatDataset`` (lists of token ids per field);
            see ``default_collater``.
        pad_seq_len_divisible: Forwarded to ``default_collater`` (round the padded length up to a
            multiple of this value).
        low: Smallest token id to draw. For the Nemotron 3 Nano tokenizer ids 0-999 are special or
            added tokens, so the default draws ordinary tokens only.
        high: One past the largest token id to draw (the vocabulary size).
        seed: Base seed. Each batch is seeded by (seed, CRC32 of its real tokens), so the draw is
            reproducible and independent of dataloader worker scheduling.

    Returns:
        ``default_collater``'s output. ``input_ids`` is a new int64 tensor of shape [batch, sequence]
        whose padding positions (``attention_mask == 0``) hold ids drawn uniformly from [low, high);
        every other field, including ``attention_mask``, ``padding_mask`` and ``labels``, is unchanged.
    """
    out = default_collater(batch, pad_seq_len_divisible=pad_seq_len_divisible)
    padding = out["attention_mask"] == 0
    input_ids = out["input_ids"].clone()
    rng = np.random.default_rng([seed, zlib.crc32(input_ids[~padding].numpy().tobytes())])
    input_ids[padding] = torch.from_numpy(rng.integers(low, high, size=int(padding.sum()), dtype=np.int64))
    out["input_ids"] = input_ids
    return out
