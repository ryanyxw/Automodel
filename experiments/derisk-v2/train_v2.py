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

"""derisk-v2 entry point: derisk-v1's probed fine-tuning recipe, plus an optional expert placement.

v2.1 (random padding tokens) needs no model change; its configs only swap the collate function.
v2.2 adds an ``expert_placement: {path: <placement.json>}`` block; the placement is applied to the
sharded model right after setup, before the optimizer creates any state.

``run_v2.sh`` puts ``experiments/derisk-v1`` on PYTHONPATH, so ``train`` below is derisk-v1's
``train.py`` (this directory has no ``train.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from expert_placement import apply_expert_placement
from train import PaddingProbeRecipe

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

logger = logging.getLogger(__name__)


class PlacedPaddingProbeRecipe(PaddingProbeRecipe):
    """derisk-v1's ``PaddingProbeRecipe`` that can relabel experts per layer before training."""

    def setup(self) -> None:
        """Build the probed recipe, then apply the ``expert_placement`` block if the config has one.

        Raises:
            RuntimeError: If an optimizer already holds state (the permutation must precede it).
        """
        super().setup()
        node = self.cfg.get("expert_placement", None)
        if node is None:
            return
        if any(len(optimizer.state) for optimizer in self.optimizer):
            raise RuntimeError("expert placement must be applied before the optimizer creates state")
        placement_path = Path(node.path)
        placement = json.loads(placement_path.read_text())
        apply_expert_placement(self.model_parts[0], placement)
        if self.dist_env.is_main:
            (self.probe_dir / "expert_placement.json").write_text(json.dumps(placement))


def main() -> None:
    """Load the YAML config (plus ``--dotted.key value`` overrides) and run the recipe."""
    cfg = parse_args_and_load_config()
    recipe = PlacedPaddingProbeRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
