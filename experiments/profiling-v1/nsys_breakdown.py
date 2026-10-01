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

"""Split each GPU's captured training step into MoE phases, from the Nsight Systems reports of ``run.sh nsys``.

For every ``runs/nsys_<config>/nsys/<config>_rank<R>.nsys-rep`` (exported to SQLite next to it on first use):

1. every kernel, memcpy and memset is attributed to the innermost ``MoE``-domain NVTX range (moe_nvtx.py) that
   was open on the CPU thread when it was launched;
2. the GPU timeline of the step (first activity start to last activity end) is cut into segments, and each
   segment goes to the highest-priority phase running in it: experts > combine > dispatch > router/shared
   expert > everything else. An idle gap goes to the phase of the next activity to start, since that is what
   the GPU is waiting for (e.g. DeepEP's host-side wait for the dispatch counts).

The phases therefore add up to the step time. ``combine`` includes the time its kernels spin waiting for the
busiest GPU, which is where expert imbalance shows up on every other GPU.

Usage::

    python experiments/profiling-v1/nsys_breakdown.py baseline random natural_visible \\
        --output experiments/profiling-v1/runs/nsys_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"
# Priority order. FSDP's NCCL collectives run on their own stream, mostly overlapped with compute, so they come last.
PHASES = ("experts", "combine", "dispatch", "router_shared", "rest", "nccl")
_RANGE = re.compile(r"^L(\d+) (dispatch|experts|combine|MoE) (fwd|recompute|bwd)")
_LOAD = re.compile(r"\| ([0-9.]+)x avg$")
_RANK = re.compile(r"_rank(\d+)\.nsys-rep$")


@dataclass(frozen=True)
class Activity:
    """One GPU activity (kernel, memcpy or memset) and the MoE range it was launched in."""

    start: int
    end: int
    name: str
    phase: str
    layer: int | None
    mode: str | None
    # Which pass of this layer and mode (0 = first micro-batch's forward, ...): the k-th ``L12 experts fwd``
    # range on every GPU is the same micro-batch, so passes can be compared across GPUs.
    ordinal: int | None


def export_sqlite(report: Path) -> Path:
    """Export ``report`` (.nsys-rep) to SQLite next to it unless that file already exists; return its path."""
    sqlite_path = report.with_suffix(".sqlite")
    if not sqlite_path.exists():
        subprocess.run(
            [
                os.environ.get("PROFILING_NSYS", "nsys"),
                "export",
                "--type",
                "sqlite",
                "--output",
                str(sqlite_path),
                str(report),
            ],
            check=True,
            capture_output=True,
        )
    return sqlite_path


def _tables(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def load_activities(sqlite_path: Path) -> tuple[list[Activity], list[float]]:
    """Read one rank's GPU activities with their MoE phase, and the load ratios in its ``experts`` range names.

    Args:
        sqlite_path: Nsight Systems SQLite export of one rank.

    Returns:
        ``(activities, expert_loads)``: activities sorted by start time; ``expert_loads`` holds the
        ``<ratio>x avg`` of every forward ``experts`` range (one per MoE layer and micro-batch).
    """
    db = sqlite3.connect(sqlite_path)
    tables = _tables(db)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    moe_domains = {
        domain_id
        for domain_id, text in db.execute("SELECT domainId, text FROM NVTX_EVENTS WHERE eventType = 75")
        if text == "MoE"
    }

    # NVTX push/pop ranges of the MoE domain, per CPU thread.
    ranges_by_thread: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for start, end, text, text_id, domain_id, tid in db.execute(
        "SELECT start, end, text, textId, domainId, globalTid FROM NVTX_EVENTS WHERE eventType = 59 AND end IS NOT NULL"
    ):
        name = text if text is not None else strings.get(text_id, "")
        if (not moe_domains or domain_id in moe_domains) and _RANGE.match(name):
            ranges_by_thread[tid].append((start, end, name))

    # CUDA API calls: which thread launched each correlation id, and when.
    launches_by_thread: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for start, tid, correlation in db.execute(
        "SELECT start, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        launches_by_thread[tid].append((start, correlation))

    # Number each range among the ranges with the same "L<layer> <phase> <mode>" on this rank, in time order.
    ordinal_of: dict[tuple[int, str], int] = {}
    seen: dict[str, int] = defaultdict(int)
    for start, _, name in sorted(r for ranges in ranges_by_thread.values() for r in ranges):
        key = _RANGE.match(name).group(0)
        ordinal_of[(start, name)] = seen[key]
        seen[key] += 1

    range_of: dict[int, tuple[str, int]] = {}
    for tid, launches in launches_by_thread.items():
        ranges = sorted(ranges_by_thread.get(tid, []), key=lambda r: (r[0], -r[1]))
        launches.sort()
        stack: list[tuple[int, int, str]] = []
        i = 0
        for start, correlation in launches:
            while i < len(ranges) and ranges[i][0] <= start:
                stack.append(ranges[i])
                i += 1
            # Push/pop ranges on one thread nest, so every closed range is above the open ones.
            while stack and stack[-1][1] < start:
                stack.pop()
            if stack:
                range_start, _, name = stack[-1]
                range_of[correlation] = (name, ordinal_of[(range_start, name)])

    queries = {
        "CUPTI_ACTIVITY_KIND_KERNEL": "SELECT start, end, correlationId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY": "SELECT start, end, correlationId, NULL FROM CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET": "SELECT start, end, correlationId, NULL FROM CUPTI_ACTIVITY_KIND_MEMSET",
    }
    activities = []
    for table, query in queries.items():
        if table not in tables:
            continue
        for start, end, correlation, name_id in db.execute(query):
            kernel = strings.get(name_id, "") if name_id is not None else table.rsplit("_", 1)[-1].lower()
            if kernel.startswith("nccl"):
                activities.append(Activity(start, end, kernel, "nccl", None, None, None))
                continue
            if correlation not in range_of:
                activities.append(Activity(start, end, kernel, "rest", None, None, None))
                continue
            range_name, ordinal = range_of[correlation]
            match = _RANGE.match(range_name)
            layer, phase, mode = int(match.group(1)), match.group(2), match.group(3)
            phase = "router_shared" if phase == "MoE" else phase
            activities.append(Activity(start, end, kernel, phase, layer, mode, ordinal))
    activities.sort(key=lambda a: a.start)

    expert_loads = [
        float(load.group(1))
        for ranges in ranges_by_thread.values()
        for _, _, name in ranges
        if " experts fwd " in name and (load := _LOAD.search(name))
    ]
    db.close()
    return activities, expert_loads


def partition(activities: list[Activity]) -> dict[str, object]:
    """Cut the step's GPU timeline into phases (see the module docstring).

    Args:
        activities: One rank's GPU activities, sorted by start time.

    Returns:
        ``{"step_s": float, "phases_s": {phase: seconds}, "layers_s": {layer: {phase: seconds}},
        "modes_s": {"<phase> <mode>": seconds}, "passes_s": {"L<layer> <mode> <ordinal>": {phase: seconds}}}``;
        ``phases_s`` sums to ``step_s``.
    """
    priority = {phase: index for index, phase in enumerate(PHASES)}
    events = []
    for index, activity in enumerate(activities):
        events.append((activity.start, 1, index))
        events.append((activity.end, -1, index))
    events.sort()
    starts = [a.start for a in activities]

    phases_s: dict[str, float] = defaultdict(float)
    layers_s: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    modes_s: dict[str, float] = defaultdict(float)
    passes_s: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    active: set[int] = set()
    next_start = 0  # index of the first activity that has not started yet
    previous = events[0][0]
    for time, kind, index in events:
        if time > previous:
            if active:
                owner = activities[min(active, key=lambda i: (priority[activities[i].phase], i))]
            else:
                while next_start < len(starts) and starts[next_start] < time:
                    next_start += 1
                owner = activities[min(next_start, len(activities) - 1)]
            seconds = (time - previous) * 1e-9
            phases_s[owner.phase] += seconds
            if owner.layer is not None:
                layers_s[owner.layer][owner.phase] += seconds
                modes_s[f"{owner.phase} {owner.mode}"] += seconds
                passes_s[f"L{owner.layer} {owner.mode} {owner.ordinal}"][owner.phase] += seconds
        if kind == 1:
            active.add(index)
        else:
            active.discard(index)
        previous = time
    step_s = (max(a.end for a in activities) - activities[0].start) * 1e-9
    return {
        "step_s": step_s,
        "phases_s": {phase: phases_s.get(phase, 0.0) for phase in PHASES},
        "layers_s": {str(layer): dict(values) for layer, values in sorted(layers_s.items())},
        "modes_s": dict(sorted(modes_s.items())),
        "passes_s": {key: dict(values) for key, values in passes_s.items()},
    }


def summarize(ranks: list[dict[str, object]]) -> dict[str, float]:
    """Split the average GPU's step into expert compute, waiting for the busiest GPU, all-to-all and the rest.

    In every MoE layer pass (one layer, one micro-batch, forward / recompute / backward), the all-to-all after the
    expert GEMMs finishes on all GPUs together, when the GPU with the most expert work is done. A GPU therefore
    waits ``max over GPUs(expert time) - its own expert time`` in that pass, inside its combine / dispatch range.

    Args:
        ranks: :func:`partition` results of all GPUs of one config.

    Returns:
        Seconds per step, averaged over GPUs: ``step``, ``experts`` (expert GEMMs), ``wait`` (waiting for the
        busiest GPU), ``all_to_all`` (dispatch + combine minus the wait), ``other`` (everything else: Mamba,
        attention, router, shared expert, loss, optimizer, FSDP). The last four add up to ``step``.
    """
    n = len(ranks)
    passes = set().union(*(r["passes_s"] for r in ranks))
    experts = wait = exchange = 0.0
    for key in passes:
        per_rank = [r["passes_s"].get(key, {}) for r in ranks]
        expert_times = [p.get("experts", 0.0) for p in per_rank]
        experts += sum(expert_times) / n
        wait += max(expert_times) - sum(expert_times) / n
        exchange += sum(p.get("combine", 0.0) + p.get("dispatch", 0.0) for p in per_rank) / n
    step = sum(r["step_s"] for r in ranks) / n
    return {
        "step": step,
        "experts": experts,
        "wait": wait,
        "all_to_all": exchange - wait,
        "other": step - experts - exchange,
    }


def analyze_config(name: str) -> list[dict[str, object]]:
    """Return one :func:`partition` result per rank (plus ``rank`` and ``expert_load_*``) for ``nsys_<name>``."""
    reports = sorted((RUNS_DIR / f"nsys_{name}" / "nsys").glob(f"{name}_rank*.nsys-rep"))
    if not reports:
        raise FileNotFoundError(f"no reports under {RUNS_DIR / f'nsys_{name}' / 'nsys'}")
    results = []
    for report in reports:
        activities, loads = load_activities(export_sqlite(report))
        result = partition(activities)
        result["rank"] = int(_RANK.search(report.name).group(1))
        result["expert_load_mean"] = sum(loads) / len(loads) if loads else None
        result["expert_load_max"] = max(loads) if loads else None
        results.append(result)
    return sorted(results, key=lambda r: r["rank"])


def main() -> None:
    """Analyze the given configs, print a per-rank table and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "nsys_breakdown.json")
    args = parser.parse_args()

    breakdown = {name: analyze_config(name) for name in args.configs}
    summaries = {name: summarize(ranks) for name, ranks in breakdown.items()}
    args.output.write_text(json.dumps({"summary": summaries, "ranks": breakdown}))
    header = f"{'config':<16}{'rank':>5}{'step s':>9}" + "".join(f"{p:>15}" for p in PHASES) + f"{'load mean':>11}"
    print(header)
    for name, ranks in breakdown.items():
        for r in ranks:
            load = r["expert_load_mean"]
            cells = "".join(f"{r['phases_s'][p]:>15.2f}" for p in PHASES)
            print(f"{name:<16}{r['rank']:>5}{r['step_s']:>9.2f}{cells}{load if load is None else f'{load:.2f}x':>11}")
    print()
    print(f"{'config':<16}" + "".join(f"{key:>12}" for key in ("step", "experts", "wait", "all_to_all", "other")))
    for name, summary in summaries.items():
        print(
            f"{name:<16}"
            + "".join(f"{summary[key]:>12.2f}" for key in ("step", "experts", "wait", "all_to_all", "other"))
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
