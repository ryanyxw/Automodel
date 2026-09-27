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

# Runtime environment for main-v1. Source it:  source experiments/main-v1/env.sh
#
# Uses the venv built by setup_env.sh, puts this checkout on PYTHONPATH, and resolves every CUDA
# library from the venv's nvidia-* wheels (CUDA 12.8). TransformerEngine loads cuDNN / NVRTC / cuRAND
# from the system first (<LIB>_HOME, then CUDA_HOME, then /usr/local/cuda) with RTLD_GLOBAL; this
# host's CUDA 12.4 toolkit ships cuDNN 8.9.4, which then interposes on TE's cuDNN 9 calls and breaks
# fused attention (CUDNN_STATUS_BAD_PARAM). The <LIB>_HOME variables below point TE at the wheels.

EXP_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXP_VENV=${EXP_VENV:-$HOME/venvs/automodel-cu12}
export EXP_REPO EXP_VENV

_exp_site=$("$EXP_VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PATH="$EXP_VENV/bin:$PATH"
export PYTHONPATH="$EXP_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDNN_HOME="$_exp_site/nvidia/cudnn"
export NVRTC_HOME="$_exp_site/nvidia/cuda_nvrtc"
export CURAND_HOME="$_exp_site/nvidia/curand"
export LD_LIBRARY_PATH="$_exp_site/nvidia/cudnn/lib:$_exp_site/nvidia/nccl/lib:$_exp_site/nvidia/cublas/lib:$_exp_site/nvidia/cuda_nvrtc/lib:$_exp_site/nvidia/cuda_runtime/lib"

# WANDB_API_KEY / HF_TOKEN are exported from ~/.bashrc, which non-interactive shells skip. Read them
# from an interactive shell without echoing them (the marker strips anything .bashrc prints).
for _exp_var in WANDB_API_KEY HF_TOKEN; do
  if [ -z "${!_exp_var:-}" ]; then
    _exp_value=$(bash -ic "printf '__EXPENV__%s' \"\${$_exp_var:-}\"" 2>/dev/null)
    _exp_value=${_exp_value##*__EXPENV__}
    [ -n "$_exp_value" ] && export "$_exp_var=$_exp_value"
  fi
done
unset _exp_site _exp_var _exp_value
