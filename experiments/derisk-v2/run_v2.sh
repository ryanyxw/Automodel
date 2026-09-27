#!/usr/bin/env bash
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

# Run one derisk-v2 config on the 8 local GPUs through derisk-v1's run_arm.sh (same busy-GPU guard,
# nvidia-smi telemetry and GPU-process log), with derisk-v2's entry point, configs and runs directory.
#
# Usage: bash experiments/derisk-v2/run_v2.sh <config_name> <run_name> [--dotted.key value ...]
#   config_name  file stem under experiments/derisk-v2/configs/
#   run_name     output directory under experiments/derisk-v2/runs/ and the wandb run name
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V1=$(cd "$HERE/../derisk-v1" && pwd)
# train_v2.py imports derisk-v1's train.py and padding_probe.py.
export PYTHONPATH="$V1${PYTHONPATH:+:$PYTHONPATH}"
export DERISK_ENTRY="$HERE/train_v2.py"
export DERISK_CONFIG_DIR="$HERE/configs"
export DERISK_RUNS_DIR="$HERE/runs"
exec bash "$V1/run_arm.sh" "$@"
