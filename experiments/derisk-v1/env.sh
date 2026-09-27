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

# Runtime environment for derisk-v1. Source it:  source experiments/derisk-v1/env.sh
#
# Uses the venv built by setup_env.sh, puts this checkout on PYTHONPATH, and resolves every CUDA
# library from the venv's nvidia-* wheels (CUDA 12.8). TransformerEngine loads cuDNN / NVRTC / cuRAND
# from the system first (<LIB>_HOME, then CUDA_HOME, then /usr/local/cuda) with RTLD_GLOBAL; this
# host's CUDA 12.4 toolkit ships cuDNN 8.9.4, which then interposes on TE's cuDNN 9 calls and breaks
# fused attention (CUDNN_STATUS_BAD_PARAM). The <LIB>_HOME variables below point TE at the wheels.

DERISK_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DERISK_VENV=${DERISK_VENV:-$HOME/venvs/automodel-cu12}
export DERISK_REPO DERISK_VENV

_derisk_site=$("$DERISK_VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PATH="$DERISK_VENV/bin:$PATH"
export PYTHONPATH="$DERISK_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDNN_HOME="$_derisk_site/nvidia/cudnn"
export NVRTC_HOME="$_derisk_site/nvidia/cuda_nvrtc"
export CURAND_HOME="$_derisk_site/nvidia/curand"
export LD_LIBRARY_PATH="$_derisk_site/nvidia/cudnn/lib:$_derisk_site/nvidia/nccl/lib:$_derisk_site/nvidia/cublas/lib:$_derisk_site/nvidia/cuda_nvrtc/lib:$_derisk_site/nvidia/cuda_runtime/lib"

# WANDB_API_KEY / HF_TOKEN are exported from ~/.bashrc, which non-interactive shells skip. Read them
# from an interactive shell without echoing them (the marker strips anything .bashrc prints).
for _derisk_var in WANDB_API_KEY HF_TOKEN; do
  if [ -z "${!_derisk_var:-}" ]; then
    _derisk_value=$(bash -ic "printf '__DERISK__%s' \"\${$_derisk_var:-}\"" 2>/dev/null)
    _derisk_value=${_derisk_value##*__DERISK__}
    [ -n "$_derisk_value" ] && export "$_derisk_var=$_derisk_value"
  fi
done
unset _derisk_site _derisk_var _derisk_value
