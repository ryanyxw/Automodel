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

# Reproduce derisk-v1 end to end: config guard, data stats, 3-step smoke tests, then the two 20-step
# arms strictly one after the other on the same 8 GPUs, then the HTML report.
#
# Usage: bash experiments/derisk-v1/run_all.sh        (SKIP_SMOKE=1 skips the smoke tests)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$DERISK_REPO"

# The arms may differ only in the padding rule and in run names / output paths.
python - "$HERE/configs/pad_batchmax.yaml" "$HERE/configs/pad_fixed8192.yaml" <<'EOF'
import sys
import yaml

allowed = {
    ("dataloader", "collate_fn", "pad_seq_len_divisible"),
    ("checkpoint", "checkpoint_dir"),
    ("wandb", "name"),
    ("wandb", "dir"),
    ("probe", "output_dir"),
}

def flatten(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from flatten(value, prefix + (key,))
    else:
        yield prefix, node

a, b = (dict(flatten(yaml.safe_load(open(path)))) for path in sys.argv[1:])
diff = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
unexpected = sorted(diff - allowed)
if unexpected:
    sys.exit(f"configs differ beyond the padding rule: {unexpected}")
print("config guard ok; differences:", sorted(diff))
EOF

python "$HERE/data_stats.py" --output "$HERE/runs/data_stats.json"

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  for arm in pad_batchmax pad_fixed8192; do
    bash "$HERE/run_arm.sh" "$arm" "smoke_$arm" --step_scheduler.max_steps 3 --wandb.group nano3-tulu3-padding-mfu-smoke
  done
fi

for arm in pad_batchmax pad_fixed8192; do
  bash "$HERE/run_arm.sh" "$arm" "$arm"
done

python "$HERE/make_report.py" --runs-dir "$HERE/runs" --output "$DERISK_REPO/hub/derisk-v1/index.html"
