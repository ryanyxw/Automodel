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

# Reproduce main-v1 end to end: config guard, CPU checks, data, 3-step smoke tests, then the 12 measured
# runs strictly one after another on the same 8 GPUs, then the HTML report.
#
# Per padding setting (fixed, then batchmax): default, then the expert placement computed from the default
# run's router counts, then random tokens, placement, natural text hidden, natural text visible, and finally
# default again, to measure how much two identical runs differ.
#
# Usage: bash experiments/main-v1/run_all.sh        (SKIP_SMOKE=1 skips the smoke tests)
# Runs that already finished (runs/<name>/exit.json with exit code 0) are skipped, so the script can be
# restarted after an interruption.
# Variants such as main-v1.1 reuse this queue through environment overrides (defaults are main-v1's):
# EXP_CONFIG_DIR, EXP_RUNS_DIR, EXP_PLACEMENTS_DIR, EXP_NOTES, EXP_REPORT, EXP_TITLE, plus run_one.sh's
# EXP_GPUS, EXP_NPROC and EXP_MASTER_PORT.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$EXP_REPO"
CONFIG_DIR=${EXP_CONFIG_DIR:-$HERE/configs}
RUNS_DIR=${EXP_RUNS_DIR:-$HERE/runs}
PLACEMENTS_DIR=${EXP_PLACEMENTS_DIR:-$HERE/placements}
export EXP_CONFIG_DIR="$CONFIG_DIR" EXP_RUNS_DIR="$RUNS_DIR"
mkdir -p "$RUNS_DIR" "$PLACEMENTS_DIR"

python "$HERE/check_configs.py" --configs "$CONFIG_DIR"
[ -f "$HERE/data/train.jsonl" ] || python "$HERE/prepare_data.py"
python "$HERE/test_main_v1.py"

finished() {
  python -c "import json, sys; sys.exit(json.load(open(sys.argv[1]))['exit_code'] != 0)" "$RUNS_DIR/$1/exit.json" 2>/dev/null
}
run() {  # run <config> <run_name> [--dotted.key value ...]
  local config=$1 name=$2
  shift 2
  if finished "$name"; then
    echo "skip $name (already finished)"
    return
  fi
  bash "$HERE/run_one.sh" "$config" "$name" "$@"
}

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  smoke=(--step_scheduler.max_steps 3 --wandb.group smoke)
  run fixed_default smoke_fixed_default "${smoke[@]}"
  python "$HERE/expert_placement.py" --iterations 2000 \
    --routing "$RUNS_DIR/smoke_fixed_default/routing.npz" --output "$RUNS_DIR/smoke_fixed_default/placement.json"
  run fixed_placement smoke_fixed_placement "${smoke[@]}" \
    --expert_placement.path "$(realpath --relative-to="$EXP_REPO" "$RUNS_DIR/smoke_fixed_default/placement.json")"
  run fixed_natural_visible smoke_fixed_natural_visible "${smoke[@]}"
  run batchmax_random smoke_batchmax_random "${smoke[@]}"
fi

for setting in fixed batchmax; do
  run "${setting}_default" "${setting}_default"
  if [ ! -f "$PLACEMENTS_DIR/$setting.json" ]; then
    python "$HERE/expert_placement.py" \
      --routing "$RUNS_DIR/${setting}_default/routing.npz" --output "$PLACEMENTS_DIR/$setting.json"
  fi
  for method in random placement natural_masked natural_visible; do
    run "${setting}_${method}" "${setting}_${method}"
  done
  run "${setting}_default" "${setting}_default_repeat"
done

python "$HERE/make_report.py" --runs-dir "$RUNS_DIR" --notes "${EXP_NOTES:-$HERE/notes.html}" \
  --output "${EXP_REPORT:-$EXP_REPO/hub/main-v1/index.html}" --title "${EXP_TITLE:-Padding and MoE training speed — main-v1}" \
  ${EXP_REFERENCE_RUNS:+--reference-runs-dir "$EXP_REFERENCE_RUNS"}
