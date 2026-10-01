# profiling-v1: why do random / visible padding make fixed-padding MoE training faster? (4 GPUs, Nsight)

derisk-v2 found that with every micro-batch padded to 8,192 tokens, filling the padding with random tokens (v2.1)
or with visible natural text (v2.3b) cuts the step time of Nemotron 3 Nano 30B-A3B full fine-tuning by 12% and 34%
on 8 GPUs. This experiment (1) checks that padded MFU still rises with these methods on 4 GPUs and (2) captures
Nsight Systems timelines annotated so the reason is visible directly: how much expert work each GPU gets in each
MoE layer, and how long the other GPUs wait for the busiest one.

| Config | Padding positions hold | Visible to attention / Mamba |
|---|---|---|
| `baseline` | `<\|im_end\|>` (AutoModel's default collate function) | no |
| `random` | random ordinary token ids (derisk-v2 `random_pad.py`) | no |
| `natural_visible` | a window of held-out Tulu-3 text (derisk-v2 `natural_pad.py`) | yes |

Setup, otherwise identical to derisk's `pad_fixed8192`: Tulu-3 (the same 2,560 conversations in the same order),
every micro-batch [2, 8192], 128 conversations per step (16 micro-batches per GPU), 20 steps, FSDP2, DeepEP,
activation checkpointing. Differences from derisk:

- **4 GPUs (4–7), expert parallelism 4** (32 experts per GPU). GPU 1's cooling failed during main-v1.
- **Plain SGD instead of Adam.** Adam's state does not fit on 4 GPUs (about 97 GiB per GPU); SGD without momentum
  has none (51 GiB peak). Forward, backward and routing are unchanged; only the short optimizer step differs.
- No wandb (`WANDB_MODE=disabled`); every metric is in `runs/<config>/probe_steps.jsonl` and `summary.json`.

## What the NVTX ranges show (`moe_nvtx.py`)

`run.sh nsys` turns on `profile.moe_nvtx`, which adds a `MoE` NVTX domain (its own row per thread in Nsight
Systems). Every MoE layer gets, in forward, activation-checkpoint recompute and backward:

| Range | What runs | Color |
|---|---|---|
| `L12 MoE fwd` / `recompute` | the whole layer: router, shared expert and the three phases below | grey |
| `L12 dispatch fwd` | DeepEP all-to-all sending each token to the GPUs hosting its experts | blue |
| `L12 experts fwd \| 212,874 tok \| 2.17x avg` | grouped GEMMs over the token-to-expert assignments this GPU received | green (< 1.25x) → yellow → orange → red (≥ 2.5x) |
| `L12 combine fwd \| 0.61x avg` | DeepEP all-to-all returning the outputs; its kernels spin until the busiest GPU is done | purple |

`x avg` is this GPU's received assignments divided by the average over the 4 GPUs (exact with fixed padding, since
every GPU sends the same number; see the module docstring). The counts come from DeepEP, which already returns
them to the host, so the ranges add no GPU synchronization.

### Reading a capture

Reports are written by Nsight Systems 2024.5 (`env.sh`), so open them with Nsight Systems 2024.5 or newer. Open the
4 reports of one config together (File → Open, select all four `*_rank*.nsys-rep`; they share one timeline), expand
each process, and look at the `MoE` NVTX row (CPU side) and the NVTX ranges projected onto its CUDA HW row (GPU side).
Zoom in on one MoE layer of one micro-batch, e.g. the first `L01 experts fwd` of the captured step (step 3):

| | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| baseline `L01 experts fwd` label | 8,430 tok, 0.09x | 68,118 tok, 0.69x | **308,935 tok, 3.14x** (red) | 7,733 tok, 0.08x |
| baseline expert GEMMs, fwd / bwd | 0.5 / 1.0 ms | 3.5 / 5.6 ms | **11.7 / 24.0 ms** | 0.4 / 0.9 ms |
| baseline all-to-all (combine fwd / dispatch bwd) | **18.2 / 41.0 ms** | 13.9 / 33.4 ms | 5.1 / 5.1 ms | **17.4 / 41.2 ms** |
| natural_visible `L01 experts fwd` label | 1.02x | 0.97x | 1.00x | 1.01x |
| natural_visible expert GEMMs, fwd / bwd | 3.2 / 6.1 ms | 3.1 / 5.7 ms | 3.2 / 6.0 ms | 3.3 / 6.0 ms |
| natural_visible all-to-all, fwd / bwd | 5.1 / 4.1 ms | 4.3 / 4.4 ms | 4.1 / 4.0 ms | 5.0 / 4.0 ms |

With `<|im_end|>` padding all pad tokens of a micro-batch go to the same 6 experts, and in layer 1 most of them live
on GPU 2: it computes for 11.7 ms while the other GPUs finish in under 4 ms and then spin in DeepEP's
`cached_notify_combine` kernel inside their purple combine range until GPU 2 is done. Expert time plus all-to-all
time is about the same on every GPU (the layer takes as long as its busiest GPU). With visible natural text every
GPU gets about the average and the waiting disappears.

**Where the wait lives:** in the all-to-all right after the expert GEMMs, i.e. `combine fwd` / `combine recompute`
and, in backward, `dispatch bwd` (the backward of the dispatch is a DeepEP combine). Ranges are labelled by the
forward phase they belong to.

## Results

### MFU runs (20 steps, steps 2–19 averaged)

| Config | Step time | Padded MFU | Useful MFU | Busiest vs average GPU (mean / worst) | Peak memory |
|---|---|---|---|---|---|
| baseline | 24.47 s | 21.8% | 1.80% | 1.84x / 3.23x | 57.3 GiB |
| random | 21.67 s (−11.4%) | 24.6% | 2.03% | 1.52x / 2.44x | 57.1 GiB |
| natural_visible | 19.18 s (−21.6%) | 27.8% | 2.30% | 1.08x / 1.21x | 54.9 GiB |

Padded MFU rises with both methods on 4 GPUs, as on 8 (derisk-v2: −12% and −34%). The mean loss is identical
(1.006) in all three: padding never reaches the loss.

### Where the time goes (`nsys_breakdown.py`, captured step 3, averaged over the 4 GPUs)

| Config | Step | Expert GEMMs | Waiting for the busiest GPU | All-to-all transfer | Everything else |
|---|---|---|---|---|---|
| baseline | 24.04 s | 6.03 s | **4.44 s** | 4.76 s | 8.82 s |
| random | 22.26 s | 5.67 s | **2.77 s** | 4.58 s | 9.24 s |
| natural_visible | 19.17 s | 4.92 s | **0.41 s** | 4.21 s | 9.64 s |

"Waiting" is, summed over every MoE layer pass (layer × micro-batch × forward / recompute / backward), the busiest
GPU's expert time minus the average GPU's. Less waiting explains 94% of random's saving (1.67 of 1.78 s) and 83% of
natural_visible's (4.03 of 4.87 s). Expert GEMMs themselves also get faster when the load is even (−1.1 s), while
"everything else" grows by 0.8 s with visible filler, mostly attention and Mamba now running over it.

Runs interrupted by a session restart or captured with the host's nsys 2023.4 (which fails to import these
traces with "Wrong event order") are kept in `runs/excluded/` and not used.

## Files

| File | Purpose |
|---|---|
| `env.sh` | Runtime environment: derisk-v1's env with the prebuilt venv and HF cache of this host, GPUs 4–7 |
| `configs/*.yaml` | The three configs; `check_configs.py` fails on any other difference |
| `profile_train.py` | Entry point: derisk-v1's `PaddingProbeRecipe` plus the MoE NVTX ranges |
| `moe_nvtx.py`, `test_moe_nvtx.py` | The NVTX ranges and their CPU tests |
| `run.sh` | `run.sh mfu <config>` (timed, 20 steps) or `run.sh nsys <config>` (step `NSYS_STEP`=3, all GPUs) |
| `run_all.sh` | Config check, 3 MFU runs, 3 captures, breakdown (skips finished runs) |
| `nsys_breakdown.py` | Per-GPU phase breakdown of the captures and the step decomposition above (`runs/nsys_breakdown.json`) |

## Reproduce

```bash
# once: test/NVTX packages next to the prebuilt venv, Nsight Systems 2024.5 (no root: dpkg -x), the datasets
# cache, and derisk-v2's natural-text filler
/home/elwin/.local/bin/uv pip install --python /home/elwin/venvs/automodel-cu12/bin/python \
    --target experiments/profiling-v1/.pydeps nvtx pytest
curl -sSLO https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb
dpkg -x nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb ~/opt/nsys-2024.5.1
source experiments/profiling-v1/env.sh
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python experiments/derisk-v2/build_filler.py \
    --output experiments/derisk-v2/filler/tulu3_train_2560_10560.npy   # also fills the datasets cache
python -m pytest experiments/profiling-v1/test_moe_nvtx.py

bash experiments/profiling-v1/run_all.sh     # about 50 min (3 x 10 min MFU runs, 3 x 4 min captures)
```
