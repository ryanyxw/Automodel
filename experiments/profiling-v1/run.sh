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

# Run one profiling-v1 config on the 4 GPUs in PROFILING_GPUS (env.sh), either as a timed MFU run or as a
# short Nsight Systems capture with the MoE NVTX ranges.
#
# Usage: bash experiments/profiling-v1/run.sh mfu  <config> [--dotted.key value ...]
#        bash experiments/profiling-v1/run.sh nsys <config> [--dotted.key value ...]
#   config  baseline | random | natural_visible
#
# mfu:  all 20 steps, no NVTX beyond the recipe's per-step ranges, nvidia-smi telemetry every 500 ms.
#       Outputs: runs/<config>/ (probe_steps.jsonl, summary.json, routing.npz, train.log, ...).
# nsys: NSYS_STEP (default 3, 0-based) captured on every GPU, with profile.moe_nvtx on; the run stops after it.
#       Outputs: runs/nsys_<config>/nsys/<config>_rank<R>.nsys-rep, written by $PROFILING_NSYS (env.sh); the trace
#       is cuda,nvtx (NSYS_TRACE overrides it).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

MODE=$1
NAME=$2
shift 2
CONFIG="$HERE/configs/$NAME.yaml"
[ -f "$CONFIG" ] || { echo "unknown config: $NAME ($CONFIG)" >&2; exit 2; }
case "$MODE" in
  mfu) RUN_NAME=$NAME ;;
  nsys) RUN_NAME=nsys_$NAME ;;
  *) echo "mode must be mfu or nsys, got $MODE" >&2; exit 2 ;;
esac
RUN_DIR="$HERE/runs/$RUN_NAME"
REL_RUN_DIR=$(realpath -m --relative-to="$DERISK_REPO" "$RUN_DIR")
NPROC=$(echo "$PROFILING_GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$RUN_DIR"

# Fair comparison: refuse to start while anything else is running on our GPUs.
busy=$(nvidia-smi -i "$PROFILING_GPUS" --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 > 5 || $3 > 2048')
if [ -n "$busy" ]; then
  echo "GPUs $PROFILING_GPUS are busy (index, utilization %, MiB used), refusing to start:" >&2
  echo "$busy" >&2
  exit 3
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DERISK_GIT_COMMIT=$(git -C "$DERISK_REPO" rev-parse HEAD)
export DERISK_RUN_NAME=$RUN_NAME

nvidia-smi -i "$PROFILING_GPUS" --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 500 > "$RUN_DIR/gpu_telemetry.csv" &
telemetry_pid=$!
trap 'kill $telemetry_pid 2>/dev/null || true' EXIT

train_args=(
  --config "$CONFIG"
  --probe.output_dir "$REL_RUN_DIR"
  --checkpoint.checkpoint_dir "$REL_RUN_DIR/checkpoint"
)
launcher=(torchrun --nproc-per-node "$NPROC" --master-port 29531)
if [ "$MODE" = nsys ]; then
  step=${NSYS_STEP:-3}
  mkdir -p "$RUN_DIR/nsys"
  ranks="[$(seq -s, 0 $((NPROC - 1)))]"
  train_args+=(
    --step_scheduler.max_steps $((step + 1))
    --profile.moe_nvtx true
    --probe.nsys_start_step "$step"
    --probe.nsys_end_step "$step"
    --probe.nsys_ranks "$ranks"
  )
  launcher+=(--no-python "$PROFILING_NSYS" profile
    --capture-range=cudaProfilerApi --capture-range-end=stop
    --trace="${NSYS_TRACE:-cuda,nvtx}" --sample=none --cpuctxsw=none
    --output "$REL_RUN_DIR/nsys/${NAME}_rank%q{RANK}" --force-overwrite=true
    python)
fi

cd "$DERISK_REPO"
echo "{\"mode\": \"$MODE\", \"config\": \"$NAME\", \"gpus\": \"$PROFILING_GPUS\", \"git_commit\": \"$DERISK_GIT_COMMIT\", \"start_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"overrides\": \"$*\"}" \
  > "$RUN_DIR/launch.json"
set +e
"${launcher[@]}" "$HERE/profile_train.py" "${train_args[@]}" "$@" 2>&1 | tee "$RUN_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e
echo "{\"end_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"exit_code\": $rc}" > "$RUN_DIR/exit.json"
exit "$rc"
