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

"""Check that every local GPU can sustain load without overheating before a timed run.

First waits (up to ``--max-wait`` seconds) until every GPU has cooled below ``--start-below`` °C, so a run that
just ended does not count against the next one. Then runs a BF16 matmul load on all GPUs at once for ``--seconds``
and samples ``nvidia-smi`` meanwhile. Fails if any GPU reports thermal slowdown in at least
``--max-thermal-share`` of the samples (a single flagged sample on a warm but healthy GPU does not count), reaches
``--max-temp``, or delivers less than ``--min-relative`` of the median GPU's throughput. During main-v1 one GPU's cooling failed mid-run: it reached 91 °C, throttled to ~1 GHz, and
slowed every step, because each MoE layer waits for the slowest GPU.

Only the GPUs in ``CUDA_VISIBLE_DEVICES`` (all GPUs when unset) are loaded and judged; results are keyed by
physical GPU index.

Usage: python experiments/main-v1/gpu_health.py [--seconds 20] [--output health.json]; exit code 0 = healthy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

# nvidia-smi clocks_event_reasons bits: 0x20 software thermal slowdown, 0x40 hardware thermal slowdown.
_THERMAL_BITS = 0x20 | 0x40


def _burn(device_index: int, seconds: float, results: dict) -> None:
    """Run back-to-back 8192³ BF16 matmuls on one CUDA device; store its TFLOP/s in ``results[device_index]``."""
    device = torch.device(f"cuda:{device_index}")
    x = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    start, count = time.time(), 0
    while time.time() - start < seconds:
        for _ in range(20):
            x @ x
        torch.cuda.synchronize(device)
        count += 20
    results[device_index] = count * 2 * 8192**3 / (time.time() - start) / 1e12


def _physical_ids() -> list[int]:
    """Physical GPU index of each visible CUDA device, in CUDA device order."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(gpu) for gpu in visible.split(",")]
    return list(range(torch.cuda.device_count()))


def _sample(physical: list[int]) -> list[dict[str, float]]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            ",".join(str(gpu) for gpu in physical),
            "--query-gpu=index,temperature.gpu,clocks.sm,clocks_event_reasons.active",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        index, temp, clock, reasons = (field.strip() for field in line.split(","))
        rows.append({"gpu": int(index), "temp": float(temp), "clock": float(clock), "reasons": int(reasons, 16)})
    return rows


def main() -> None:
    """Load every GPU, then report and judge per-GPU throughput and temperature."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--max-temp", type=float, default=85.0)
    parser.add_argument("--min-relative", type=float, default=0.85)
    parser.add_argument("--max-thermal-share", type=float, default=0.25)
    parser.add_argument("--start-below", type=float, default=50.0)
    parser.add_argument("--max-wait", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    physical = _physical_ids()
    waited = time.time()
    while time.time() - waited < args.max_wait and max(row["temp"] for row in _sample(physical)) >= args.start_below:
        time.sleep(5)
    waited = time.time() - waited
    manager = mp.Manager()
    results = manager.dict()
    workers = [mp.Process(target=_burn, args=(device, args.seconds, results)) for device in range(len(physical))]
    for worker in workers:
        worker.start()
    peak = {gpu: {"temp": 0.0, "min_clock": 1e9, "thermal_samples": 0, "samples": 0} for gpu in physical}
    while any(worker.is_alive() for worker in workers):
        for row in _sample(physical):
            state = peak[row["gpu"]]
            state["temp"] = max(state["temp"], row["temp"])
            state["min_clock"] = min(state["min_clock"], row["clock"])
            state["thermal_samples"] += bool(row["reasons"] & _THERMAL_BITS)
            state["samples"] += 1
        time.sleep(2)
    for worker in workers:
        worker.join()

    tflops = {physical[device]: value for device, value in results.items()}
    median = sorted(tflops.values())[len(tflops) // 2]
    problems = []
    for gpu in physical:
        state = peak[gpu]
        if state["thermal_samples"] >= args.max_thermal_share * max(state["samples"], 1):
            problems.append(f"GPU {gpu}: thermal slowdown in {state['thermal_samples']} of {state['samples']} samples")
        if state["temp"] >= args.max_temp:
            problems.append(f"GPU {gpu}: reached {state['temp']:.0f} °C")
        if tflops.get(gpu, 0.0) < args.min_relative * median:
            problems.append(f"GPU {gpu}: {tflops.get(gpu, 0.0):.0f} TFLOP/s vs median {median:.0f}")
    report = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "waited_to_cool_s": round(waited),
        "tflops": {str(gpu): round(value, 1) for gpu, value in sorted(tflops.items())},
        "peak": {str(gpu): state for gpu, state in peak.items()},
        "problems": problems,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=1))
    logger.info("GPU health: %s", json.dumps(report))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
