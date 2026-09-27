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

"""Fail if a derisk-v2 config differs from its derisk-v1 base anywhere except where intended.

Usage: python experiments/derisk-v2/check_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_V1_CONFIGS = _HERE.parent / "derisk-v1" / "configs"
# Run names and output paths may always differ.
_ALWAYS_ALLOWED = (
    ("checkpoint", "checkpoint_dir"),
    ("wandb", "project"),
    ("wandb", "group"),
    ("wandb", "name"),
    ("wandb", "dir"),
    ("probe", "output_dir"),
)
# v2 config -> (derisk-v1 base config, key prefixes that are the intended change).
PAIRS = {
    "v21_pad_fixed8192_randpad": ("pad_fixed8192", (("dataloader", "collate_fn"),)),
    "v21_pad_batchmax_randpad": ("pad_batchmax", (("dataloader", "collate_fn"),)),
    "v22_pad_fixed8192_placement": ("pad_fixed8192", (("expert_placement",),)),
    "v22_pad_batchmax_placement": ("pad_batchmax", (("expert_placement",),)),
    "v23a_pad_fixed8192_natural_masked": ("pad_fixed8192", (("dataloader", "collate_fn"),)),
    "v23b_pad_fixed8192_natural_visible": ("pad_fixed8192", (("dataloader", "collate_fn"),)),
    "v23a_pad_batchmax_natural_masked": ("pad_batchmax", (("dataloader", "collate_fn"),)),
    "v23b_pad_batchmax_natural_visible": ("pad_batchmax", (("dataloader", "collate_fn"),)),
}


def _flatten(node: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(node, dict):
        out: dict[tuple[str, ...], object] = {}
        for key, value in node.items():
            out.update(_flatten(value, prefix + (str(key),)))
        return out
    return {prefix: node}


def main() -> None:
    """Compare every v2 config with its base and exit non-zero on an unintended difference."""
    failures = []
    for name, (base, intended) in PAIRS.items():
        v2 = _flatten(yaml.safe_load((_HERE / "configs" / f"{name}.yaml").read_text()))
        v1 = _flatten(yaml.safe_load((_V1_CONFIGS / f"{base}.yaml").read_text()))
        allowed = _ALWAYS_ALLOWED + intended
        diff = sorted(key for key in v1.keys() | v2.keys() if v1.get(key) != v2.get(key))
        unexpected = [key for key in diff if not any(key[: len(prefix)] == prefix for prefix in allowed)]
        print(f"{name}: differs from {base} in {['.'.join(k) for k in diff]}")
        if unexpected:
            failures.append(f"{name}: unexpected differences {['.'.join(k) for k in unexpected]}")
    if failures:
        sys.exit("\n".join(failures))
    print("config guard ok")


if __name__ == "__main__":
    main()
