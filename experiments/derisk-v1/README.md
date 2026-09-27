# derisk-v1: does padding change MFU? (Nemotron 3 Nano SFT on Tulu-3)

Two otherwise identical 20-step full-parameter SFT runs of
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` on 8× H100 (FSDP2 + expert parallelism EP=8, DeepEP,
TE attention, activation checkpointing), on the same random 2,560 conversations from
`allenai/tulu-3-sft-mixture`, run one after the other. Only the padding rule differs:

| Arm | Config | Padding |
|---|---|---|
| `pad_batchmax` | `configs/pad_batchmax.yaml` | each micro-batch padded to its own longest sequence |
| `pad_fixed8192` | `configs/pad_fixed8192.yaml` | every micro-batch padded to 8,192 tokens (2× the 4,096 truncation limit) |

The report is written to `hub/derisk-v1/index.html` (single self-contained file). Metrics are also in
wandb, project `derisk-v1`, group `nano3-tulu3-padding-mfu` (smoke tests and diagnostics in their own groups).

## What is measured

- **Padded MFU**: AutoModel's NemotronH FLOPs formula on each micro-batch's padded shape (identical to
  AutoModel's built-in `mfu`), so pad tokens count as work.
- **Effective MFU**: the same formula on each sequence's real length, so only real tokens count.
- Both over wall-clock step time and over GPU-synchronized step time; plus step time, real / padded
  tokens per second, pad fraction, loss, per-rank peak memory, GPU telemetry.
- **Router hooks** (`padding_probe.py`) on all 23 MoE layers: top-6 assignment counts per micro-batch,
  layer, expert, split into real tokens and pad tokens bucketed by distance from the end of the text.
  Nemotron V3 does not pass its padding mask to the MoE, so pad tokens are really dispatched to these
  experts. Activation-checkpoint recomputation is filtered out.

## Files

| File | Purpose |
|---|---|
| `setup_env.sh` | Builds the uv venv this host needs (see "Environment" below) |
| `env.sh` | Runtime environment; `source` it before running anything |
| `configs/*.yaml` | The two arms; `run_all.sh` fails if they differ beyond padding and output paths |
| `train.py` | Entry point: `PaddingProbeRecipe`, a subclass of the standard fine-tuning recipe that adds the probes |
| `padding_probe.py` | Router hooks and routing summaries |
| `data_stats.py` | Token lengths of the subset and predicted padding per arm (CPU only) |
| `bench_shape_overhead.py` | One-GPU microbenchmark of per-new-sequence-length costs (cuDNN attention plans, CCE autotune) |
| `run_arm.sh` | Runs one arm with GPU telemetry and a GPU-process log |
| `run_all.sh` | Config guard, data stats, smoke tests, both arms, report |
| `profile_nsys.sh` | Nsight Systems capture of steady-state steps, one report per rank |
| `make_report.py`, `report_template.html` | HTML report generator |
| `patches/deepep-intranode-only.patch` | DeepEP build patch for this host (intranode-only, no HybridEP) |

## Reproduce

```bash
bash experiments/derisk-v1/setup_env.sh        # once: uv venv + sm_90 extension builds (~20 min)
bash experiments/derisk-v1/run_all.sh          # everything, including the report
# one arm:  bash experiments/derisk-v1/run_arm.sh pad_fixed8192 pad_fixed8192
# report:   source experiments/derisk-v1/env.sh && \
#           python experiments/derisk-v1/make_report.py --runs-dir experiments/derisk-v1/runs --output hub/derisk-v1/index.html
```

## Outputs per run (`runs/<run_name>/`)

| File | Contents |
|---|---|
| `probe_steps.jsonl` | Every metric per step, plus per-rank compute time and peak memory |
| `summary.json` | Mean / std / min / max over steps ≥ 2 |
| `routing.npz` | `counts` [steps, micro-batches, layers, token categories, experts], `routing_weight_sums`, `microbatch_shapes` [steps, ranks, micro-batches, (batch, padded length, real tokens)], `layer_ids`, `metadata` |
| `run_metadata.json` | Package versions, env vars (no secrets), GPU list, `nvidia-smi topo -m`, MoE layout |
| `resolved_config.yaml` | Config after CLI overrides |
| `gpu_telemetry.csv`, `gpu_processes.csv` | nvidia-smi samples (500 ms) and every GPU process (5 s) during the run |
| `train.log`, `launch.json`, `exit.json`, `checkpoint/training.jsonl`, `wandb/` | Logs and AutoModel's own metrics |

## Profiling later with Nsight Systems

`bash experiments/derisk-v1/profile_nsys.sh <arm> [start_step] [end_step] [ranks]` reruns an arm with
AutoModel's module-level NVTX ranges and `cudaProfilerStart/Stop` around the chosen steps. The recipe
always emits `train_step<N>` and `microbatch<i>_B<batch>xS<seq>` NVTX ranges. Use `routing.npz`
(`hot_rank` per layer is derivable from `counts`) to pick the GPU that hosts the pad-token experts; it
is the straggler in the DeepEP combine. The host nsys is 2023.4; with this stack its importer only
handles `cuda,nvtx` tracing (the default; set `NSYS_TRACE` to change it with a newer nsys). Sample traces
of pad-fixed8192 (steps 2–3, ranks 0–1) and their kernel summaries are in `runs/nsys_pad_fixed8192/nsys/`.

## Environment

This host runs NVIDIA driver 550 (CUDA 12.4). `uv.lock` pins torch 2.10.0+cu130 and the NGC
nemo-automodel 26.08 container is CUDA 13.3; neither runs here. `setup_env.sh` keeps the lockfile
versions but uses PyPI's torch 2.10.0 (CUDA 12.8) and compiles TransformerEngine's PyTorch extension,
causal-conv1d, mamba-ssm, and DeepEP for sm_90. `env.sh` points TransformerEngine at the venv's cuDNN 9
instead of the host toolkit's cuDNN 8.9.4. `pyproject.toml` and `uv.lock` are untouched.
