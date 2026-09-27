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

"""Build the derisk-v2.3 filler stream: natural Tulu-3 text from outside the training subset (CPU only).

The training subset is ``train[:2560]`` of tulu-3-sft-mixture after ``shuffle(seed=42)``. The filler takes the
next conversations of that same shuffled order (``train[2560:10560]`` by default, so it is disjoint), formats
them exactly like the training data (Nemotron chat template via ``ChatDataset``), and concatenates their
token ids into one int32 stream that ``natural_pad.py`` samples windows from.

Usage::

    python experiments/derisk-v2/build_filler.py --output experiments/derisk-v2/filler/tulu3_train_2560_10560.npy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig

logger = logging.getLogger(__name__)

_BASE_CONFIG = Path(__file__).resolve().parents[1] / "derisk-v1" / "configs" / "pad_fixed8192.yaml"


def main() -> None:
    """Tokenize the filler conversations and save the concatenated token stream."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="train[2560:10560]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(_BASE_CONFIG.read_text())
    fields = {key: value for key, value in cfg["dataset"].items() if key != "_target_"}
    if fields["split"] != "train[:2560]":
        raise ValueError(f"expected the training subset to be train[:2560], got {fields['split']}")
    fields["split"] = args.split
    tokenizer = NeMoAutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_name_or_path"], trust_remote_code=True)
    dataset = ChatDatasetConfig(**fields).build(tokenizer=tokenizer)
    stream = np.concatenate([np.asarray(dataset[i]["input_ids"], dtype=np.int32) for i in range(len(dataset))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, stream)
    ids, counts = np.unique(stream, return_counts=True)
    top = np.argsort(counts)[::-1][:10]
    logger.info(
        "filler: %d conversations, %d tokens, %d distinct ids; top-10 ids cover %.1f%% (%s)",
        len(dataset),
        stream.size,
        ids.size,
        100 * counts[top].sum() / stream.size,
        [tokenizer.convert_ids_to_tokens(int(i)) for i in ids[top]],
    )


if __name__ == "__main__":
    main()
