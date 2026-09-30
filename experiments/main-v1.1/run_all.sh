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

# Reproduce main-v1.1 end to end: main-v1's queue (guards, CPU checks, smoke tests, the 12 measured runs one after
# another, report) with main-v1.1's LoRA configs, on GPUs 4-7 with 4 processes.
#
# Usage: bash experiments/main-v1.1/run_all.sh     (SKIP_SMOKE=1 skips the smoke tests; EXP_GPUS picks other GPUs)
# Finished runs are skipped, so the script can be restarted after an interruption.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V1=$(cd "$HERE/../main-v1" && pwd)
source "$V1/env.sh"

python "$HERE/check_configs.py"
export EXP_CONFIG_DIR="$HERE/configs" EXP_RUNS_DIR="$HERE/runs" EXP_PLACEMENTS_DIR="$HERE/placements"
export EXP_GPUS=${EXP_GPUS:-4,5,6,7} EXP_NPROC=4 EXP_MASTER_PORT=29521
export EXP_NOTES="$HERE/notes.html" EXP_REPORT="$EXP_REPO/hub/main-v1.1/index.html"
export EXP_TITLE="Padding and MoE training speed, LoRA on 4 GPUs — main-v1.1" EXP_REFERENCE_RUNS="$V1/runs"
exec bash "$V1/run_all.sh"
