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

Runs a BF16 matmul load on all GPUs at once for ``--seconds`` and samples ``nvidia-smi`` meanwhile. Fails if any
GPU reports thermal slowdown, reaches ``--max-temp``, or delivers less than ``--min-relative`` of the median
GPU's throughput. During main-v1 one GPU's cooling failed mid-run: it reached 91 °C, throttled to ~1 GHz, and
slowed every step, because each MoE layer waits for the slowest GPU.

Usage: python experiments/main-v1/gpu_health.py [--seconds 20] [--output health.json]; exit code 0 = healthy.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

# nvidia-smi clocks_event_reasons bits: 0x20 software thermal slowdown, 0x40 hardware thermal slowdown.
_THERMAL_BITS = 0x20 | 0x40


def _burn(gpu: int, seconds: float, results: dict) -> None:
    """Run back-to-back 8192³ BF16 matmuls on one GPU and store its TFLOP/s in ``results[gpu]``."""
    device = torch.device(f"cuda:{gpu}")
    x = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    start, count = time.time(), 0
    while time.time() - start < seconds:
        for _ in range(20):
            x @ x
        torch.cuda.synchronize(device)
        count += 20
    results[gpu] = count * 2 * 8192**3 / (time.time() - start) / 1e12


def _sample() -> list[dict[str, float]]:
    out = subprocess.run(
        [
            "nvidia-smi",
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    n_gpus = torch.cuda.device_count()
    manager = mp.Manager()
    results = manager.dict()
    workers = [mp.Process(target=_burn, args=(gpu, args.seconds, results)) for gpu in range(n_gpus)]
    for worker in workers:
        worker.start()
    peak = {gpu: {"temp": 0.0, "min_clock": 1e9, "thermal": False} for gpu in range(n_gpus)}
    while any(worker.is_alive() for worker in workers):
        for row in _sample():
            state = peak[row["gpu"]]
            state["temp"] = max(state["temp"], row["temp"])
            state["min_clock"] = min(state["min_clock"], row["clock"])
            state["thermal"] = state["thermal"] or bool(row["reasons"] & _THERMAL_BITS)
        time.sleep(2)
    for worker in workers:
        worker.join()

    tflops = dict(results)
    median = sorted(tflops.values())[len(tflops) // 2]
    problems = []
    for gpu in range(n_gpus):
        state = peak[gpu]
        if state["thermal"]:
            problems.append(f"GPU {gpu}: thermal slowdown")
        if state["temp"] >= args.max_temp:
            problems.append(f"GPU {gpu}: reached {state['temp']:.0f} °C")
        if tflops.get(gpu, 0.0) < args.min_relative * median:
            problems.append(f"GPU {gpu}: {tflops.get(gpu, 0.0):.0f} TFLOP/s vs median {median:.0f}")
    report = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
