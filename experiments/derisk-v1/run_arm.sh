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

# Run one derisk-v1 arm on the 8 local GPUs, with nvidia-smi telemetry sampled alongside.
#
# Usage: bash experiments/derisk-v1/run_arm.sh <arm> <run_name> [--dotted.key value ...]
#   arm       pad_batchmax | pad_fixed8192 (selects configs/<arm>.yaml)
#   run_name  output directory under experiments/derisk-v1/runs/ and the wandb run name
# Extra arguments are forwarded as config overrides, e.g. --step_scheduler.max_steps 3
# Later derisk versions reuse this script by setting DERISK_ENTRY (training script), DERISK_CONFIG_DIR
# and DERISK_RUNS_DIR; the defaults are derisk-v1's.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

ARM=$1
RUN_NAME=$2
shift 2
ENTRY=${DERISK_ENTRY:-$HERE/train.py}
CONFIG="${DERISK_CONFIG_DIR:-$HERE/configs}/$ARM.yaml"
RUN_DIR="${DERISK_RUNS_DIR:-$HERE/runs}/$RUN_NAME"
[ -f "$CONFIG" ] || { echo "unknown arm: $ARM ($CONFIG)" >&2; exit 2; }
mkdir -p "$RUN_DIR"
REL_RUN_DIR=$(realpath --relative-to="$DERISK_REPO" "$RUN_DIR")

# Fair comparison: refuse to start while anything else is doing real work on the GPUs. This is a
# shared machine, so a small idle CUDA context (e.g. another user's notebook, < 2 GiB, GPUs idle) is
# recorded rather than fatal; gpu_processes.csv logs every GPU process for the whole run.
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader,nounits)
heavy=$(echo "$apps" | awk -F', ' 'NF >= 3 && $3 >= 2048')
max_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
if [ -n "$heavy" ] || [ "${max_util:-0}" -gt 5 ]; then
  echo "GPUs are busy (max utilization ${max_util}%), refusing to start:" >&2
  echo "$apps" >&2
  exit 3
fi
echo "$apps" > "$RUN_DIR/gpu_processes_at_start.txt"

export DERISK_GIT_COMMIT=$(git -C "$DERISK_REPO" rev-parse HEAD)
export DERISK_RUN_NAME="$RUN_NAME"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
# Both arms: limit allocator fragmentation (the pad-token expert hot spot drives large, uneven
# grouped-GEMM allocations on the GPU that hosts those experts).
export PYTORCH_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 500 > "$RUN_DIR/gpu_telemetry.csv" &
telemetry_pid=$!
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory,gpu_uuid --format=csv,nounits -l 5 \
  > "$RUN_DIR/gpu_processes.csv" &
processes_pid=$!
trap 'kill $telemetry_pid $processes_pid 2>/dev/null || true' EXIT

cat > "$RUN_DIR/launch.json" <<EOF
{"arm": "$ARM", "run_name": "$RUN_NAME", "config": "$(realpath --relative-to="$DERISK_REPO" "$CONFIG")", "entry": "$(realpath --relative-to="$DERISK_REPO" "$ENTRY")",
 "overrides": "$*", "git_commit": "$DERISK_GIT_COMMIT", "host": "$(hostname)",
 "start_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "venv": "$DERISK_VENV"}
EOF

cd "$DERISK_REPO"
set +e
torchrun --nproc-per-node 8 --master-port 29511 "$ENTRY" \
  --config "$CONFIG" \
  --probe.output_dir "$REL_RUN_DIR" \
  --checkpoint.checkpoint_dir "$REL_RUN_DIR/checkpoint" \
  --wandb.dir "$REL_RUN_DIR" \
  --wandb.name "$RUN_NAME" \
  "$@" 2>&1 | tee "$RUN_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e
echo "{\"end_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"exit_code\": $rc}" > "$RUN_DIR/exit.json"
exit "$rc"
