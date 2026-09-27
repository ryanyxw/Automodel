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

"""derisk-v2.3 collate function: fill padding with a window of natural text instead of one repeated token.

Each padded row gets a random contiguous window of natural Tulu-3 text (``build_filler.py``) in its padding
positions; labels stay -100, so the filler never enters the loss.

* ``visible=False`` (v2.3a): the filler keeps attention mask 0. Attention and Mamba still ignore it, so each
  filler token reaches the router with only its own embedding, like v2.1's random tokens but with natural
  token frequencies.
* ``visible=True`` (v2.3b): the filler gets attention mask 1, so it is processed like a continuation of the
  row and its tokens reach the router with real context. With causal attention the real tokens, which come
  first, never see the filler, so their outputs and the loss are unchanged. ``real_mask`` records which
  positions are real so derisk's probe and effective MFU still count only real tokens.
"""

from __future__ import annotations

import functools
import zlib

import numpy as np
import torch

from nemo_automodel.components.datasets.utils import default_collater


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
        filler_path: ``.npy`` file with the 1-D int32 filler token stream from ``build_filler.py``.
        visible: Whether the filler is visible to attention and Mamba (attention mask 1) or masked (0).
        pad_seq_len_divisible: Forwarded to ``default_collater``.
        seed: Base seed. Each batch is seeded by (seed, CRC32 of its real tokens), so windows are reproducible
            and independent of dataloader worker scheduling.

    Returns:
        ``default_collater``'s output plus ``real_mask``: ``input_ids`` is a new int64 tensor of shape [batch,
        sequence] whose padding positions hold a contiguous filler window per row; ``real_mask`` is the
        original ``attention_mask`` (int64 [batch, sequence], right-padded, 1 for real tokens); ``labels`` is
        unchanged. With ``visible`` set, ``attention_mask`` is all ones and ``padding_mask`` all False.

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
