# main-v1: does what fills the padding change MoE training speed? (Nemotron 3 Nano SFT on IF-Chat)

Full-parameter SFT of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` on 8× H100 (FSDP2 + expert parallelism
over 8 GPUs, DeepEP, TE attention, activation checkpointing), on the same 5,120 random conversations from
`nvidia/Nemotron-Instruction-Following-Chat-v1` (split `chat_if`, truncated to 8,192 tokens), 128
conversations per step. Two padding settings × five padding methods, one run at a time on the same GPUs:

| Setting | Micro-batches padded to | Steps |
|---|---|---|
| `fixed` | 8,192 tokens | 40 (one pass over the data) |
| `batchmax` | their own longest conversation | 80 (two passes: the first includes one-time setup per new length) |

| Method | Padding positions hold | Visible to attention / Mamba |
|---|---|---|
| `default` | `<|im_end|>` (AutoModel's default collate function) | no |
| `random` | random ordinary token ids (`pad_fill.random_pad_collater`) | no |
| `placement` | `<|im_end|>`; every MoE layer's experts are reassigned to GPUs to even out the work, from the default run's router counts on the same data (`expert_placement.py`) | no |
| `natural_masked` | a random window of IF-Chat text from conversations outside the training set (`pad_fill.natural_pad_collater`) | no |
| `natural_visible` | the same windows | yes (after the real tokens; attention is causal, so real tokens never see them) |

Padding never carries labels, so every method trains on the same loss. The default run of each setting is
repeated at the end of that setting's queue to measure run-to-run noise. The report is
`hub/main-v1/index.html`; metrics are in wandb, project `main-v1` (groups `fixed`, `batchmax`, `smoke`).

## Status (2026-09-27): paused; 8 of 12 measured runs complete, a ninth half done

- Done: `fixed_*` (all five methods and `fixed_default_repeat`), `batchmax_default`, `batchmax_random`, and the first
  pass of `batchmax_placement`.
- GPU 1's cooling failed at 02:58 UTC during `batchmax_placement`'s second pass (85–91 °C, about 60% throughput).
  That pass is left out of the report, and the run is kept in `runs/excluded/` once the queue resumes. The
  interrupted `batchmax_natural_masked` is also in `runs/excluded/`.
- `experiments/main-v1.1` repeats the whole matrix with LoRA on GPUs 4–7 (full-parameter training does not fit on 4 GPUs).
- Still to run: `batchmax_placement` (both passes again), `batchmax_natural_masked`, `batchmax_natural_visible`,
  `batchmax_default_repeat` (about 1.5 h).
- To resume once every GPU passes `python experiments/main-v1/gpu_health.py`, move `runs/batchmax_placement` to
  `runs/excluded/`, then run `SKIP_SMOKE=1 bash experiments/main-v1/run_all.sh` (it skips finished runs).
- `runs/` (the repo's `.gitignore`) and the sampled data in `data/` are not committed. `prepare_data.py` recreates
  the data exactly (pinned dataset revision and seed); `data/data_stats.json` is committed because the report reads it.

## Files

| File | Purpose |
|---|---|
| `setup_env.sh`, `patches/` | Builds the uv venv this host needs (driver 550; see "Environment") |
| `env.sh` | Runtime environment; `source` it before running anything |
| `prepare_data.py` | Samples the 5,120 training conversations and the natural-text filler (`data/`) |
| `make_chat_template.py`, `templates/` | Nano's chat template with `{% generation %}` markers, verified against the stock one |
| `configs/*.yaml` | One config per (setting, method); `check_configs.py` fails on any unintended difference |
| `train.py` | Entry point: `PaddingStudyRecipe`, the standard fine-tuning recipe plus probes and expert placement |
| `padding_probe.py` | Router hooks (which experts every real / padding token is sent to) and routing summaries |
| `pad_fill.py` | Collate functions for the random-token and natural-text methods |
| `expert_placement.py` | Computes and applies the per-layer expert placement |
| `test_main_v1.py` | CPU checks of the above |
| `gpu_health.py` | 20 s full-load check of every GPU (thermal slowdown, temperature, throughput); `run_one.sh` refuses to start if it fails |
| `run_one.sh` | Runs one config with a busy-GPU guard, the GPU health check, GPU telemetry and a GPU-process log |
| `run_all.sh` | Checks, smoke tests, the 12 measured runs in order, report (skips finished runs) |
| `profile_nsys.sh` | Nsight Systems capture of chosen steps, one report per GPU |
| `make_report.py`, `report_template.html` | HTML report generator |

## Reproduce

```bash
bash experiments/main-v1/setup_env.sh          # once: uv venv + sm_90 extension builds (~20 min)
bash experiments/main-v1/run_all.sh            # everything (~4 h of GPU time), including the report
# one run:  bash experiments/main-v1/run_one.sh fixed_random fixed_random
# report:   source experiments/main-v1/env.sh && python experiments/main-v1/make_report.py
```

## Outputs per run (`runs/<run_name>/`)

| File | Contents |
|---|---|
| `probe_steps.jsonl` | Every metric per step, plus per-GPU compute time and peak memory |
| `summary.json` | Mean / std / min / max over steps ≥ 2 |
| `routing.npz` | `counts` [steps, micro-batches, MoE layers, token categories, experts] (category 0 = real tokens, 1.. = padding by distance from the text), `routing_weight_sums`, `microbatch_shapes` [steps, GPUs, micro-batches, (batch, padded length, real tokens)], `layer_ids`, `metadata` |
| `expert_placement.json` | The placement applied (placement runs only) |
| `run_metadata.json`, `resolved_config.yaml` | Package versions, env vars (no secrets), GPU list and topology, MoE layout; config after CLI overrides |
| `gpu_telemetry.csv`, `gpu_processes.csv` | nvidia-smi samples (500 ms) and every GPU process (5 s) during the run |
| `train.log`, `launch.json`, `exit.json`, `wandb/` | Logs |

## Profiling with Nsight Systems

`bash experiments/main-v1/profile_nsys.sh <config> [start_step] [end_step] [ranks]` reruns a config with
AutoModel's module-level NVTX ranges and `cudaProfilerStart/Stop` around the chosen steps (the recipe always
emits `train_step<N>` and `microbatch<i>_B<batch>xS<seq>` ranges). The host nsys is 2023.4; with this stack
its importer only handles `cuda,nvtx` tracing (the default; set `NSYS_TRACE` with a newer nsys).

## Environment

This host runs NVIDIA driver 550 (CUDA 12.4); `uv.lock`'s torch 2.10.0+cu130 and the NGC nemo-automodel
26.08 container (CUDA 13.3) need a newer driver. `setup_env.sh` keeps the lockfile versions except PyPI torch
2.10.0 (CUDA 12.8) and cuDNN 9.18.1.3, and compiles TransformerEngine 2.19, causal-conv1d, mamba-ssm and
DeepEP (intranode only, `patches/deepep-intranode-only.patch`) for sm_90. `env.sh` points TransformerEngine
at the wheel copies of cuDNN / NVRTC / cuRAND, because the host CUDA 12.4 ships cuDNN 8.9.4.

## Gotchas

- `ChatDataset` rejects multi-turn conversations with reasoning under Nano's stock chat template (the
  template drops earlier turns' reasoning, so the prefix-based loss mask fails). `templates/nano3_generation.jinja`
  fixes this without changing the rendered text.
- `ChatDataset` splits JSONL files with `str.splitlines()`, which also breaks on U+2028 / U+2029 / U+0085
  inside JSON strings, so `data/*.jsonl` are written with ASCII escapes.
- On 2026-09-27 at 02:58 UTC one GPU's cooling failed mid-run (85–91 °C under load, about 60% of normal
  throughput). Each MoE layer waits for the slowest GPU, so the whole step slowed. `make_report.py` leaves out
  every step that overlaps a thermal-slowdown sample in `gpu_telemetry.csv`. Affected runs are kept in
  `runs/excluded/` and were repeated once the GPU passed `gpu_health.py` again.
- Nemotron V3 does not pass the padding mask to its MoE layers: padding tokens are routed and computed by
  the experts like real tokens. That is what makes the padding methods matter.
