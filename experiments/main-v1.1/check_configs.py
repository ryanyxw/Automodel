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

"""Fail if a main-v1.1 config differs from main-v1's config of the same name beyond what main-v1.1 changes.

main-v1.1 changes exactly: the ``peft`` block (LoRA), ``distributed.ep_size``, the ``optimizer`` block, the wandb
project and tags, and output paths. The comparisons between main-v1.1's own configs are checked by main-v1's
``check_configs.py --configs experiments/main-v1.1/configs``.

Usage: python experiments/main-v1.1/check_configs.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_V1_CONFIGS = _HERE.parent / "main-v1" / "configs"
_ALLOWED = (
    ("peft",),
    ("distributed", "ep_size"),
    ("optimizer",),
    ("wandb", "project"),
    ("wandb", "tags"),
    ("wandb", "dir"),
    ("checkpoint", "checkpoint_dir"),
    ("probe", "output_dir"),
    ("expert_placement", "path"),
)


def _flatten(node: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(node, dict):
        out: dict[tuple[str, ...], object] = {}
        for key, value in node.items():
            out.update(_flatten(value, prefix + (str(key),)))
        return out
    return {prefix: node}


def main() -> None:
    """Compare every main-v1.1 config with main-v1's and exit non-zero on an unintended difference."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = []
    names = sorted(path.stem for path in (_HERE / "configs").glob("*.yaml"))
    if names != sorted(path.stem for path in _V1_CONFIGS.glob("*.yaml")):
        failures.append("main-v1.1 and main-v1 do not have the same set of configs")
    for name in names:
        config = _flatten(yaml.safe_load((_HERE / "configs" / f"{name}.yaml").read_text()))
        base = _flatten(yaml.safe_load((_V1_CONFIGS / f"{name}.yaml").read_text()))
        diff = sorted(key for key in base.keys() | config.keys() if base.get(key) != config.get(key))
        unexpected = [key for key in diff if not any(key[: len(prefix)] == prefix for prefix in _ALLOWED)]
        if unexpected:
            failures.append(f"{name}: unexpected differences from main-v1 {['.'.join(key) for key in unexpected]}")
        for key in (("checkpoint", "checkpoint_dir"), ("wandb", "dir"), ("probe", "output_dir")):
            if f"main-v1.1/runs/{name}" not in str(config.get(key)):
                failures.append(f"{name}: {'.'.join(key)} = {config.get(key)} is not under main-v1.1/runs/{name}")
        if config.get(("distributed", "ep_size")) != 4 or ("peft", "_target_") not in config:
            failures.append(f"{name}: expected LoRA and ep_size 4")
        logger.info("%s: differs from main-v1 in %s", name, sorted({key[0] for key in diff}))
    if failures:
        sys.exit("\n".join(failures))
    logger.info("main-v1.1 config guard ok")


if __name__ == "__main__":
    main()
