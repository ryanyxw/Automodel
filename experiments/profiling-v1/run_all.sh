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

# Config check, the three MFU runs, the three nsys captures, then the nsys breakdown, one at a time.
# Runs that already finished (exit.json with exit code 0) are skipped, so the queue can be resumed.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
python "$HERE/check_configs.py"

CONFIGS=(baseline random natural_visible)
for mode in mfu nsys; do
  for name in "${CONFIGS[@]}"; do
    run_name=$name
    [ "$mode" = nsys ] && run_name=nsys_$name
    if grep -q '"exit_code": 0' "$HERE/runs/$run_name/exit.json" 2>/dev/null; then
      echo "skip $run_name (done)"
      continue
    fi
    echo "$(date -u +%FT%TZ) start $run_name"
    bash "$HERE/run.sh" "$mode" "$name" > "$HERE/runs/$run_name.launcher.log" 2>&1
    echo "$(date -u +%FT%TZ) done $run_name"
  done
done
python "$HERE/nsys_breakdown.py" "${CONFIGS[@]}"
