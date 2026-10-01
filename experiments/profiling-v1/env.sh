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

# Runtime environment for profiling-v1. Source it:  source experiments/profiling-v1/env.sh
#
# Reuses derisk-v1's environment (CUDA 12.8 venv, cuDNN / NVRTC / cuRAND from the wheels) with the venv
# and Hugging Face cache already built on this host by another user, read-only:
#   PROFILING_VENV     venv built by experiments/derisk-v1/setup_env.sh (default: /home/elwin/venvs/automodel-cu12)
#   PROFILING_HF_HUB   Hugging Face hub cache holding the model and Tulu-3 (default: /home/elwin/.cache/huggingface/hub)
# Everything this run writes (datasets arrow cache, remote-code modules) goes to $PROFILING_HF_HOME.
# The runs use the four GPUs in PROFILING_GPUS (default 4,5,6,7: one NUMA node, and not GPU 1, whose
# cooling failed during main-v1).

PROFILING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DERISK_VENV=${PROFILING_VENV:-/home/elwin/venvs/automodel-cu12}
source "$PROFILING_DIR/../derisk-v1/env.sh"

# derisk-v1 (train.py, padding_probe.py) and derisk-v2 (random_pad.py, natural_pad.py) are imported from
# their own directories; .pydeps holds the `nvtx` package (colored NVTX ranges), installed by setup.sh.
export PYTHONPATH="$PROFILING_DIR:$PROFILING_DIR/.pydeps:$DERISK_REPO/experiments/derisk-v1:$DERISK_REPO/experiments/derisk-v2:$PYTHONPATH"
export HF_HOME=${PROFILING_HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${PROFILING_HF_HUB:-/home/elwin/.cache/huggingface/hub}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PROFILING_GPUS=${PROFILING_GPUS:-4,5,6,7}
export CUDA_VISIBLE_DEVICES=$PROFILING_GPUS
export WANDB_MODE=disabled
# The host's nsys 2023.4 cannot import these captures ("Wrong event order" on CUDA API events), so run.sh uses a
# newer Nsight Systems unpacked (dpkg -x) from NVIDIA's debian11 CUDA repo; see the README.
export PROFILING_NSYS=${PROFILING_NSYS:-$HOME/opt/nsys-2024.5.1/opt/nvidia/nsight-systems/2024.5.1/bin/nsys}
