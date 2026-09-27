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

"""Fail if a main-v1 config differs from its reference config anywhere except where intended.

Usage: python experiments/main-v1/check_configs.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIGS = Path(__file__).resolve().parent / "configs"
# Run names and output paths may always differ.
_ALWAYS_ALLOWED = (
    ("checkpoint", "checkpoint_dir"),
    ("wandb", "name"),
    ("wandb", "dir"),
    ("probe", "output_dir"),
)
_COLLATE = (("dataloader", "collate_fn"),)
# config -> (reference config, key prefixes that are the intended change).
PAIRS = {
    "batchmax_default": (
        "fixed_default",
        (("dataloader", "collate_fn", "pad_seq_len_divisible"), ("step_scheduler", "max_steps"), ("wandb", "group")),
    ),
    **{
        f"{setting}_{method}": (f"{setting}_default", allowed)
        for setting in ("fixed", "batchmax")
        for method, allowed in (
            ("random", _COLLATE),
            ("placement", (("expert_placement",),)),
            ("natural_masked", _COLLATE),
            ("natural_visible", _COLLATE),
        )
    },
}


def _flatten(node: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(node, dict):
        out: dict[tuple[str, ...], object] = {}
        for key, value in node.items():
            out.update(_flatten(value, prefix + (str(key),)))
        return out
    return {prefix: node}


def main() -> None:
    """Compare every config with its reference and exit non-zero on an unintended difference."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = []
    for name, (reference, intended) in PAIRS.items():
        config = _flatten(yaml.safe_load((_CONFIGS / f"{name}.yaml").read_text()))
        base = _flatten(yaml.safe_load((_CONFIGS / f"{reference}.yaml").read_text()))
        allowed = _ALWAYS_ALLOWED + intended
        diff = sorted(key for key in base.keys() | config.keys() if base.get(key) != config.get(key))
        unexpected = [key for key in diff if not any(key[: len(prefix)] == prefix for prefix in allowed)]
        logger.info("%s: differs from %s in %s", name, reference, [".".join(key) for key in diff])
        if unexpected:
            failures.append(f"{name}: unexpected differences {['.'.join(key) for key in unexpected]}")
        stem = f"runs/{name}"
        for key in ("checkpoint", "checkpoint_dir"), ("wandb", "dir"), ("probe", "output_dir"):
            if stem not in str(config.get(key)):
                failures.append(f"{name}: {'.'.join(key)} = {config.get(key)} does not point at {stem}")
        if config.get(("wandb", "name")) != name:
            failures.append(f"{name}: wandb.name is {config.get(('wandb', 'name'))}")
    if failures:
        sys.exit("\n".join(failures))
    logger.info("config guard ok")


if __name__ == "__main__":
    main()
