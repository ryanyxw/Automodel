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

"""Fail unless the profiling-v1 configs differ only in the collate function and their run name / output paths.

Usage: python experiments/profiling-v1/check_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
NAMES = ("baseline", "random", "natural_visible")
_ALLOWED = {("dataloader", "collate_fn"), ("checkpoint", "checkpoint_dir"), ("probe", "output_dir")}


def _flatten(node: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    """Flatten nested dicts into {key path: leaf value}."""
    if not isinstance(node, dict):
        return {prefix: node}
    flat: dict[tuple[str, ...], object] = {}
    for key, value in node.items():
        flat.update(_flatten(value, (*prefix, str(key))))
    return flat


def _differences(name: str) -> list[str]:
    """Return the unexpected differences between ``configs/<name>.yaml`` and ``configs/baseline.yaml``."""
    base = _flatten(yaml.safe_load((CONFIG_DIR / "baseline.yaml").read_text()))
    other = _flatten(yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text()))
    problems = []
    for key in sorted(set(base) | set(other)):
        if key[:2] in _ALLOWED or base.get(key) == other.get(key):
            continue
        problems.append(f"{name}: {'.'.join(key)} = {other.get(key)!r} (baseline {base.get(key)!r})")
    for section, field in (("checkpoint", "checkpoint_dir"), ("probe", "output_dir")):
        expected = f"experiments/profiling-v1/runs/{name}"
        value = str(other.get((section, field), ""))
        if not value.startswith(expected):
            problems.append(f"{name}: {section}.{field} = {value!r} does not start with {expected!r}")
    return problems


def main() -> None:
    """Print every unexpected difference and exit 1 if there is any."""
    problems = [problem for name in NAMES for problem in _differences(name)]
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"configs OK: {', '.join(NAMES)} differ only in the collate function and run paths")


if __name__ == "__main__":
    main()
