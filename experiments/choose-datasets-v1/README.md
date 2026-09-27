# choose-datasets-v1: which natural SFT dataset makes random pad tokens pay off under batchmax?

derisk-v2 found that filling padding with random token ids (instead of `<|im_end|>`) cuts step time by 12% when
every micro-batch is padded to 8,192 tokens, but by only ~1% when each micro-batch is padded to its own longest
sequence ("batchmax") on Tulu-3. This experiment screens candidate datasets for the batchmax setting (Nemotron 3 Nano
30B-A3B, 8× H100, FSDP2 + EP=8, 2 sequences per GPU, global batch 128) and picks the ones worth a GPU A/B.

**Phase 1 (done): CPU screen.** Sample each candidate from the Hub, tokenize it exactly as the recipe does, replay
derisk's data order, and predict EP imbalance and step time with `<|im_end|>` vs random pads using a model calibrated
on the derisk-v1/v2 runs. Report: `hub/choose-datasets-v1/index.html`.

**Phase 2 (not run, needs approval):** A/B of `<|im_end|>` vs random pads on the top two datasets at truncation
8,192, two passes × 20 steps each (derisk-v2's protocol).

## Files

| File | Purpose |
|---|---|
| `sample_datasets.py` | Draws ~6,000 random rows per candidate from the Hub parquet files (random row groups across all files) and writes `samples/<name>.jsonl` (OpenAI `messages`) plus `samples/<name>.meta.json` (split-wide composition, what was read) |
| `make_generation_template.py` | Writes `templates/nano3_generation.jinja`: Nano's chat template with `{% generation %}` markers, needed for multi-turn reasoning rows (IF-Chat); checks identical rendering and loss masks on the samples |
| `ep_model.py` | Sampler replay and the calibrated EP-imbalance / step-time model (fit from derisk-v1/v2 `routing.npz` and `probe_steps.jsonl`) |
| `test_ep_model.py` | CPU checks: replay on a hand example, replay = derisk's recorded micro-batch shapes, model reproduces the calibration runs and Tulu's measured imbalance |
| `screen.py` | Tokenizes every dataset with `ChatDataset` (untruncated, 4,096, 8,192), screens 20 random 2,560-sample orders for 2 and 4 sequences per GPU; writes `runs/screen.json` and `runs/lengths/<name>.npz` |
| `make_report.py`, `notes.html` | HTML report (reuses derisk-v1's `report_template.html`) and its interpretation text |

## Candidates

| Name | Hub dataset | Why |
|---|---|---|
| `tulu3` | `allenai/tulu-3-sft-mixture` (seed-42 shuffle, first 6,000 rows) | derisk-v1/v2 baseline; its first 2,560 rows are exactly derisk's subset |
| `nemotron_ifchat_v1` | `nvidia/Nemotron-Instruction-Following-Chat-v1`, `chat_if` | Nemotron 3 Nano's own SFT data; mixed reasoning on/off, multi-turn |
| `mixture_of_thoughts` | `open-r1/Mixture-of-Thoughts`, `all` | long reasoning traces (math, code, science) mixed with shorter ones |
| `wildchat_1m` | `allenai/WildChat-1M` | real, heavy-tailed chat |

Not screened: `nvidia/Nemotron-Post-Training-Dataset-v2` (AutoModel's Nano-30B example mix) is gated and this HF
account has not accepted its terms.

## Reproduce

```bash
source experiments/derisk-v1/env.sh
python experiments/choose-datasets-v1/sample_datasets.py --output-dir experiments/choose-datasets-v1/samples  # ~0.6 GB of row groups
python experiments/choose-datasets-v1/make_generation_template.py
python experiments/choose-datasets-v1/test_ep_model.py
python experiments/choose-datasets-v1/screen.py --workers 64          # ~1 min on this host
python experiments/choose-datasets-v1/make_report.py --output hub/choose-datasets-v1/index.html
```

## Gotchas found on the way

- `ChatDataset` rejects multi-turn conversations with reasoning under Nano's stock chat template (the template drops
  earlier turns' `<think>`, so the prefix-based loss mask fails). Use `templates/nano3_generation.jinja`
  (`dataset.chat_template: experiments/choose-datasets-v1/templates/nano3_generation.jinja`).
- `ChatDataset` reads JSONL with `str.splitlines()`, which also splits on U+2028 / U+2029 / U+0085 inside JSON
  strings. Write JSONL ASCII-escaped (`json.dumps` default), as `sample_datasets.py` does.
- WildChat-1M's parquet files are ordered by time (and therefore model), so its sample's model mix is off by up to
  6 points; IF-Chat and Mixture-of-Thoughts are shuffled and their samples match the splits within 0.3 points.
