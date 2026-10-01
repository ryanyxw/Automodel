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

"""profiling-v1 entry point: derisk-v1's probed fine-tuning recipe, plus per-phase MoE NVTX ranges.

Named ``profile_train.py`` so it does not shadow derisk-v1's ``train.py``, which it imports.

``run.sh`` launches it (``env.sh`` puts ``experiments/derisk-v1`` and ``experiments/derisk-v2`` on
PYTHONPATH, so ``train`` below is derisk-v1's ``train.py``). With ``profile.moe_nvtx: true`` (set by
``run.sh nsys``) every MoE layer gets the NVTX ranges of :mod:`moe_nvtx`; the MFU runs leave it off.
"""

from __future__ import annotations

import logging

from moe_nvtx import install_moe_nvtx
from train import PaddingProbeRecipe

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

logger = logging.getLogger(__name__)


class ProfiledPaddingProbeRecipe(PaddingProbeRecipe):
    """derisk-v1's ``PaddingProbeRecipe`` that can annotate every MoE layer with NVTX ranges."""

    def setup(self) -> None:
        """Build the probed recipe, then install the MoE NVTX ranges if ``profile.moe_nvtx`` is set."""
        super().setup()
        node = self.cfg.get("profile", None)
        if node is not None and bool(node.get("moe_nvtx", False)):
            layers = install_moe_nvtx(self.model_parts[0])
            logger.info("MoE NVTX ranges on %d layers: %s", len(layers), layers)


def main() -> None:
    """Load the YAML config (plus ``--dotted.key value`` overrides) and run the recipe."""
    cfg = parse_args_and_load_config()
    recipe = ProfiledPaddingProbeRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
