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

"""main-v1 collate functions that change what the padding positions of a micro-batch hold.

Both start from ``default_collater`` (right padding with ``<|im_end|>``, attention mask 0, label -100) and
only rewrite the padding positions, so the real tokens, their labels, and therefore the loss are unchanged.

* :func:`random_pad_collater` ("random tokens"): each padding position gets a random ordinary token id. The
  model still ignores padding in attention and Mamba, so every pad token reaches the MoE router with only
  its own embedding; with one repeated pad token they all look identical and go to the same 6 experts.
* :func:`natural_pad_collater` ("natural text"): each padded row gets a random contiguous window of IF-Chat
  text from conversations outside the training subset (``prepare_data.py``). With ``visible=False`` the
  filler stays hidden from attention and Mamba, like the random tokens. With ``visible=True`` it gets
  attention mask 1, so it is read as a continuation of the row and its tokens reach the router with
  context. Attention is causal and the filler comes after the real tokens, so real tokens never see it;
  ``real_mask`` records which positions are real for the measurements in ``train.py``.
"""

from __future__ import annotations

import functools
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
    """Collate like ``default_collater``, then overwrite the padding positions of ``input_ids`` with random ids.

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


@functools.lru_cache(maxsize=4)
def _filler_stream(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def natural_pad_collater(
    batch: list[dict[str, list[int] | torch.Tensor]],
    *,
    filler_path: str,
    visible: bool,
    pad_seq_len_divisible: int | None = None,
    seed: int = 1234,
) -> dict[str, torch.Tensor]:
    """Collate like ``default_collater``, then fill each row's padding with a window of natural text.

    Args:
        batch: Per-sample dictionaries as produced by ``ChatDataset``; see ``default_collater``.
        filler_path: ``.npy`` file with the 1-D int32 filler token stream written by ``prepare_data.py``.
        visible: Whether the filler is visible to attention and Mamba (attention mask 1) or hidden (0).
        pad_seq_len_divisible: Forwarded to ``default_collater``.
        seed: Base seed. Each batch is seeded by (seed, CRC32 of its real tokens), so windows are reproducible
            and independent of dataloader worker scheduling.

    Returns:
        ``default_collater``'s output plus ``real_mask``: ``input_ids`` is a new int64 tensor of shape [batch,
        sequence] whose padding positions hold one contiguous filler window per row; ``real_mask`` is the
        original ``attention_mask`` (int64 [batch, sequence], right-padded, 1 for real tokens); ``labels`` is
        unchanged (-100 on the filler). With ``visible`` set, ``attention_mask`` is all ones and
        ``padding_mask`` all False.

    Raises:
        ValueError: If a row needs a longer window than the filler stream holds.
    """
    out = default_collater(batch, pad_seq_len_divisible=pad_seq_len_divisible)
    real_mask = out["attention_mask"].clone()
    stream = _filler_stream(filler_path)
    input_ids = out["input_ids"].clone()
    rng = np.random.default_rng([seed, zlib.crc32(input_ids[real_mask.bool()].numpy().tobytes())])
    seq_len = input_ids.shape[1]
    for row, real_len in enumerate(real_mask.sum(dim=1).tolist()):
        n_pad = seq_len - int(real_len)
        if n_pad == 0:
            continue
        if n_pad > len(stream):
            raise ValueError(f"filler stream has {len(stream)} tokens, row needs {n_pad}")
        start = int(rng.integers(0, len(stream) - n_pad + 1))
        input_ids[row, int(real_len) :] = torch.from_numpy(np.asarray(stream[start : start + n_pad], dtype=np.int64))
    out["input_ids"] = input_ids
    out["real_mask"] = real_mask
    if visible:
        out["attention_mask"] = torch.ones_like(out["attention_mask"])
        out["padding_mask"] = torch.zeros_like(out["padding_mask"])
    return out
