# main-v1.1: main-v1 with LoRA on 4 GPUs

main-v1 (full-parameter fine-tuning of Nemotron 3 Nano 30B-A3B on 8 GPUs) was paused when one GPU's cooling
failed. Full-parameter training does not fit on the 4 healthy GPUs of the node: weights, gradients and Adam
state take about 10 bytes per parameter, i.e. about 79 GiB per GPU for 31.6B parameters on 4 GPUs, before any
activations. main-v1.1 therefore repeats the whole main-v1 matrix with LoRA on GPUs 4–7 (one NUMA node, away
from the faulty GPU 1).

Same as main-v1: the 5,120 IF-Chat conversations in the same order (main-v1's `data/`), 128 conversations per
step, 2 per GPU per micro-batch, the two padding settings (fixed 8,192 tokens, 40 steps; longest in batch, 80
steps), the five padding methods, the repeated default run, the measurements, the GPU health check, the recipe
(`experiments/main-v1/train.py`) and the collate functions.

Different (`check_configs.py` enforces that nothing else differs from main-v1's configs):

| | main-v1 | main-v1.1 |
|---|---|---|
| Training | full-parameter | LoRA, rank 8, alpha 32 (5.7M trainable weights, 0.02%) on the attention, Mamba input and shared-expert linear layers; the 128 routed experts per layer stay frozen but still compute forward and backward. Settings from AutoModel's Nano V3 LoRA example, plus `lm_head` excluded because the fused loss reads its weight directly |
| GPUs | 8 (0–7) | 4 (4–7) |
| Experts per GPU | 16 (split over 8 GPUs) | 32 (split over 4 GPUs) |
| Micro-batches per GPU per step | 8 | 16 |
| Optimizer | TE FusedAdam with master weights | torch Adam on the LoRA weights |

main-v1.1's numbers are not comparable one to one with main-v1's: both the training type and the expert
split change. The comparisons within main-v1.1 (each method against the default at the same steps) are.

## Files

| File | Purpose |
|---|---|
| `configs/*.yaml` | One config per (setting, method), generated from main-v1's |
| `check_configs.py` | Fails if a config differs from main-v1's beyond LoRA, expert split, optimizer and paths |
| `run_all.sh` | main-v1's queue with these configs on GPUs 4–7 (4 processes), then the report |
| `run_one.sh` | One run on GPUs 4–7 through main-v1's `run_one.sh` |
| `notes.html` | Report text; `experiments/main-v1/make_report.py` fills in the numbers |
| `placements/` | Expert placements computed from this experiment's default runs |

## Reproduce

```bash
bash experiments/main-v1/setup_env.sh      # once, if the venv does not exist yet
source experiments/main-v1/env.sh && python experiments/main-v1/prepare_data.py   # once, if main-v1/data is empty
bash experiments/main-v1.1/run_all.sh      # everything, including hub/main-v1.1/index.html
```

Run outputs go to `runs/` (not committed). Metrics are also in wandb, project `main-v1.1`.
