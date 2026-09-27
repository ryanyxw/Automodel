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

# Capture Nsight Systems timelines of steady-state steps for one derisk-v1 arm, one report per rank.
#
# Usage: bash experiments/derisk-v1/profile_nsys.sh <arm> [start_step] [end_step] [ranks]
#   arm         pad_batchmax | pad_fixed8192
#   start_step  first 0-based step to capture (default 8, well past warmup)
#   end_step    last 0-based step to capture, inclusive (default 9)
#   ranks       Python list of ranks that capture (default [0,1,2,3,4,5,6,7])
#
# Same config as the MFU runs, plus AutoModel's module-level NVTX ranges (--nvtx true), which cost
# host time and are therefore off in the MFU runs. The recipe always emits NVTX ranges
# train_step<N> and microbatch<i>_B<batch>xS<seq>; cudaProfilerStart/Stop bracket the window.
# Reports land in runs/nsys_<arm>/nsys/<arm>_rank<R>.nsys-rep. The host nsys is 2023.4: with this
# CUDA 12.8 / cuDNN 9.18 stack its importer fails on osrt/cublas/cudnn traces, so the default trace is
# cuda,nvtx (override with NSYS_TRACE=... when using a newer nsys). Open reports with 2023.4 or newer.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

ARM=$1
START=${2:-8}
END=${3:-9}
RANKS=${4:-[0,1,2,3,4,5,6,7]}
RUN_NAME="nsys_$ARM"
REL_RUN_DIR="experiments/derisk-v1/runs/$RUN_NAME"
mkdir -p "$DERISK_REPO/$REL_RUN_DIR/nsys"

cd "$DERISK_REPO"
torchrun --nproc-per-node 8 --master-port 29512 --no-python \
  nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace="${NSYS_TRACE:-cuda,nvtx}" --sample=none --cpuctxsw=none \
  --output "$REL_RUN_DIR/nsys/${ARM}_rank%q{RANK}" --force-overwrite=true \
  python "$HERE/train.py" \
  --config "$HERE/configs/$ARM.yaml" \
  --step_scheduler.max_steps $((END + 1)) \
  --nvtx true \
  --probe.nsys_start_step "$START" \
  --probe.nsys_end_step "$END" \
  --probe.nsys_ranks "$RANKS" \
  --probe.output_dir "$REL_RUN_DIR" \
  --checkpoint.checkpoint_dir "$REL_RUN_DIR/checkpoint" \
  --wandb.dir "$REL_RUN_DIR" \
  --wandb.name "$RUN_NAME" \
  --wandb.group nano3-tulu3-padding-mfu-nsys 2>&1 | tee "$DERISK_REPO/$REL_RUN_DIR/train.log"
