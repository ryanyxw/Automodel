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

"""derisk-v1 entry point: the standard LLM fine-tuning recipe plus padding / MFU probes.

Launch one arm at a time (run_all.sh runs both, sequentially)::

    torchrun --nproc-per-node 8 experiments/derisk-v1/train.py \
        --config experiments/derisk-v1/configs/pad_batchmax.yaml

The training math is untouched. :class:`PaddingProbeRecipe` only adds measurement:

* padded MFU (FLOPs of every micro-batch's padded [batch, sequence] shape, which is what AutoModel's
  built-in ``mfu`` reports) and effective MFU (FLOPs of the real tokens only, summed per sequence),
  each over both the wall-clock step time and a GPU-synchronized step time;
* router hooks on every MoE layer (:class:`padding_probe.RoutingProbe`);
* NVTX ranges per step and micro-batch, an optional ``cudaProfilerStart``/``Stop`` window for nsys, and
  a metadata dump (versions, topology, resolved config) for later profiling.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from padding_probe import ProbeConfig, RoutingProbe

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.utils.flops_utils import calculate_mfu
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.shared.import_utils import safe_import

_HAS_WANDB, wandb = safe_import("wandb")

logger = logging.getLogger(__name__)

_VERSIONED_PACKAGES = (
    "torch",
    "triton",
    "transformers",
    "transformer-engine",
    "transformer-engine-cu12",
    "transformer-engine-torch",
    "deep_ep",
    "mamba-ssm",
    "causal-conv1d",
    "datasets",
    "torchdata",
    "numpy",
    "wandb",
    "nvidia-nccl-cu12",
    "nvidia-cudnn-cu12",
)
_ENV_PREFIXES = ("CUDA", "NCCL", "TORCH", "PYTORCH", "NVTE", "OMP_", "HF_", "LOCAL_", "WORLD", "MASTER_", "DERISK")
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
_SUMMARY_KEYS = (
    "mfu",
    "probe/mfu_padded_wall",
    "probe/mfu_effective_wall",
    "probe/mfu_padded_compute",
    "probe/mfu_effective_compute",
    "probe/tflops_per_gpu_padded_wall",
    "probe/tflops_per_gpu_effective_wall",
    "probe/step_time_wall_s",
    "probe/step_time_compute_s",
    "probe/real_tokens_per_s_per_gpu",
    "probe/padded_tokens_per_s_per_gpu",
    "probe/pad_fraction",
    "probe/mem_reserved_peak_gib_max_rank",
    "probe/mem_allocated_peak_gib_max_rank",
    "routing/pad_assignment_fraction",
    "routing/ep_imbalance_all_max",
    "routing/ep_imbalance_all_mean",
    "routing/ep_imbalance_real_mean",
    "routing/ep_imbalance_pad_mean",
    "routing/pad_entropy_norm_mean",
    "routing/real_entropy_norm_mean",
    "loss",
    "grad_norm",
)


class PaddingProbeRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """``TrainFinetuneRecipeForNextTokenPrediction`` with the derisk-v1 measurement probes."""

    def setup(self) -> None:
        """Build the standard recipe, then attach the probes described by the ``probe:`` config block.

        Raises:
            ValueError: If the config has no ``probe:`` block or the model has no FLOPs formula.
        """
        super().setup()
        node = self.cfg.get("probe", None)
        if node is None:
            raise ValueError("derisk-v1 configs need a `probe:` section")
        self.probe_config = ProbeConfig(
            output_dir=str(node.output_dir),
            pad_distance_edges=tuple(int(edge) for edge in node.get("pad_distance_edges", (1, 8, 64, 512, 4096))),
            summary_warmup_steps=int(node.get("summary_warmup_steps", 2)),
            nsys_start_step=int(node.get("nsys_start_step", -1)),
            nsys_end_step=int(node.get("nsys_end_step", -1)),
            nsys_ranks=tuple(int(rank) for rank in node.get("nsys_ranks", (0,))),
        )
        if self.mfu_calculator.get_flops((1, 1)) is None:
            raise ValueError(f"no FLOPs formula for {type(self.mfu_calculator.config).__name__}")
        self.probe_dir = Path(self.probe_config.output_dir)
        self.probe = self.probe_config.build(
            model=self.model_parts[0], ep_size=self.moe_mesh["ep"].size(), device=self.dist_env.device
        )
        self._probe_group = self._get_dp_group(include_cp=True)
        self._probe_records: list[dict[str, object]] = []
        self._nsys_active = False
        if self.dist_env.is_main:
            self.probe_dir.mkdir(parents=True, exist_ok=True)
            _write_run_metadata(self.probe_dir, self.cfg.to_yaml_dict(use_orig_values=True), self.probe)

    def run_train_validation_loop(self) -> None:
        """Run the standard loop, then write probe outputs even if training stops early."""
        try:
            super().run_train_validation_loop()
        finally:
            self._finalize_probe()

    def _forward_backward_step(
        self,
        idx: int,
        batch: dict[str, torch.Tensor],
        *,
        loss_buffer: list[torch.Tensor],
        num_label_tokens: int,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        """Run one micro-batch under an NVTX range, telling the routing probe which tokens are padding.

        Args:
            idx: 0-based micro-batch index within the optimizer step.
            batch: Collated micro-batch on the CPU; ``input_ids`` and ``attention_mask`` are tensors of
                shape [batch, sequence], right-padded. The real tokens are given by :meth:`_real_token_mask`.
                An optional ``real_mask`` entry is removed before the model sees the batch.
            loss_buffer: Per-micro-batch detached losses, appended to by the parent recipe.
            num_label_tokens: Supervised tokens in the whole optimizer step (all ranks).
            num_batches: Micro-batches in the optimizer step.
            is_train: Whether this is a training (forward + backward) micro-batch.
        """
        if is_train:
            self.probe.begin_microbatch(idx, self._real_token_mask(batch))
        model_batch = {key: value for key, value in batch.items() if key != "real_mask"}
        batch_size, seq_len = batch["input_ids"].shape
        with torch.cuda.nvtx.range(f"microbatch{idx}_B{batch_size}xS{seq_len}"):
            super()._forward_backward_step(
                idx,
                model_batch,
                loss_buffer=loss_buffer,
                num_label_tokens=num_label_tokens,
                num_batches=num_batches,
                is_train=is_train,
            )

    def _run_train_optim_step(
        self, batches: list[dict[str, torch.Tensor]], max_grad_norm: float | None = None
    ) -> MetricsSample:
        """Run one optimizer step of the parent recipe and add probe metrics to its ``MetricsSample``.

        Args:
            batches: The step's collated micro-batches on the CPU; ``input_ids`` and ``attention_mask``
                are tensors of shape [batch, sequence], right-padded, ``attention_mask`` 1 for real tokens.
            max_grad_norm: Gradient clipping norm, forwarded to the parent recipe.

        Returns:
            The parent's metrics sample with ``probe/*`` and ``routing*/*`` entries added.
        """
        step = self.step_scheduler.step
        rank = dist.get_rank()
        if step == self.probe_config.nsys_start_step and rank in self.probe_config.nsys_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            self._nsys_active = True

        local_counts = self._local_flops_and_tokens(batches)
        torch.cuda.synchronize()
        previous_timestamp = self.timestamp
        start = time.perf_counter()
        with torch.cuda.nvtx.range(f"train_step{step}"):
            sample = super()._run_train_optim_step(batches, max_grad_norm)
        torch.cuda.synchronize()
        compute_seconds = time.perf_counter() - start
        wall_seconds = self.timestamp - previous_timestamp

        if self._nsys_active and step == self.probe_config.nsys_end_step:
            torch.cuda.cudart().cudaProfilerStop()
            self._nsys_active = False

        # Everything below runs outside the timed region (it lands in the next step's wall time).
        totals = self._dp_allreduce(local_counts, include_cp=True)
        per_rank = torch.tensor(
            [compute_seconds, torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30],
            dtype=torch.float64,
            device=self.dist_env.device,
        )
        gathered = torch.empty(
            (dist.get_world_size(self._probe_group), 3), dtype=torch.float64, device=self.dist_env.device
        )
        dist.all_gather_into_tensor(gathered, per_rank, group=self._probe_group)
        gathered = gathered.cpu()
        routing_metrics = self.probe.end_step(step, self._probe_group)
        torch.cuda.reset_peak_memory_stats()

        effective_tflops, padded_tflops = totals[0].item() / 1e12, totals[1].item() / 1e12
        real_tokens, padded_tokens = totals[2].item(), totals[3].item()
        compute_max = gathered[:, 0].max().item()
        world = self.dist_env.world_size
        peak = self.mfu_calculator.reference_mfu
        metrics = {
            "probe/mfu_padded_wall": calculate_mfu(padded_tflops, world, wall_seconds, reference_mfu=peak),
            "probe/mfu_effective_wall": calculate_mfu(effective_tflops, world, wall_seconds, reference_mfu=peak),
            "probe/mfu_padded_compute": calculate_mfu(padded_tflops, world, compute_max, reference_mfu=peak),
            "probe/mfu_effective_compute": calculate_mfu(effective_tflops, world, compute_max, reference_mfu=peak),
            "probe/tflops_per_gpu_padded_wall": padded_tflops / (world * wall_seconds),
            "probe/tflops_per_gpu_effective_wall": effective_tflops / (world * wall_seconds),
            "probe/step_time_wall_s": wall_seconds,
            "probe/step_time_compute_s": compute_max,
            "probe/step_time_compute_min_rank_s": gathered[:, 0].min().item(),
            "probe/tokens_padded": padded_tokens,
            "probe/tokens_real": real_tokens,
            "probe/pad_fraction": 1.0 - real_tokens / padded_tokens,
            "probe/real_tokens_per_s_per_gpu": real_tokens / (world * wall_seconds),
            "probe/padded_tokens_per_s_per_gpu": padded_tokens / (world * wall_seconds),
            "probe/mem_allocated_peak_gib_max_rank": gathered[:, 1].max().item(),
            "probe/mem_reserved_peak_gib_max_rank": gathered[:, 2].max().item(),
            "probe/mem_reserved_peak_gib_min_rank": gathered[:, 2].min().item(),
            **routing_metrics,
        }
        sample.metrics.update(metrics)
        if self.dist_env.is_main:
            # The parent's metrics can hold 0-d tensors (e.g. grad_norm); store plain floats.
            record = {
                "step": sample.step,
                "timestamp": sample.timestamp,
                **{
                    key: value.item() if isinstance(value, torch.Tensor) else value
                    for key, value in sample.metrics.items()
                },
            }
            record["per_rank_compute_s"] = gathered[:, 0].tolist()
            record["per_rank_mem_allocated_peak_gib"] = gathered[:, 1].tolist()
            record["per_rank_mem_reserved_peak_gib"] = gathered[:, 2].tolist()
            self._probe_records.append(record)
        return sample

    @staticmethod
    def _real_token_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return which positions of a micro-batch hold real (non-padding) tokens.

        Args:
            batch: Collated micro-batch; ``attention_mask`` and the optional ``real_mask`` are tensors of
                shape [batch, sequence] with 1 for real tokens. ``real_mask`` exists when padding positions
                are made visible to attention (derisk-v2.3), so ``attention_mask`` no longer marks them.

        Returns:
            ``real_mask`` if present, else ``attention_mask``: a tensor of shape [batch, sequence],
            right-padded, 1 for real tokens and 0 for padding.
        """
        return batch["real_mask"] if "real_mask" in batch else batch["attention_mask"]

    def _local_flops_and_tokens(self, batches: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Count this rank's model FLOPs and tokens for one optimizer step.

        Args:
            batches: The step's micro-batches on the CPU; ``input_ids`` and ``attention_mask`` are tensors
                of shape [batch, sequence], right-padded, ``attention_mask`` 1 for real tokens.

        Returns:
            Float64 tensor of shape [4]: effective FLOPs (AutoModel's FLOPs formula applied to each
            sequence's real length and summed), padded FLOPs (the formula applied to each micro-batch's
            padded [batch, sequence] shape, as AutoModel's ``mfu`` does), real tokens, padded tokens.
        """
        effective = padded = real = total = 0.0
        for batch in batches:
            batch_size, seq_len = batch["input_ids"].shape
            lengths = [int(n) for n in self._real_token_mask(batch).sum(dim=1).tolist()]
            padded += float(self.mfu_calculator.get_flops((batch_size, seq_len)))
            effective += sum(float(self.mfu_calculator.get_flops((1, n))) for n in lengths)
            real += sum(lengths)
            total += batch_size * seq_len
        return torch.tensor([effective, padded, real, total], dtype=torch.float64)

    def _finalize_probe(self) -> None:
        """Stop profiling, remove hooks, and write / upload probe outputs (rank 0 writes files)."""
        if self._nsys_active:
            torch.cuda.cudart().cudaProfilerStop()
            self._nsys_active = False
        self.probe.close()
        if not self.dist_env.is_main:
            return
        routing_path = self.probe_dir / "routing.npz"
        self.probe.save(routing_path)
        steps_path = self.probe_dir / "probe_steps.jsonl"
        with steps_path.open("w") as f:
            for record in self._probe_records:
                f.write(json.dumps(record) + "\n")
        summary = _summarize_records(self._probe_records, self.probe_config.summary_warmup_steps)
        summary_path = self.probe_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("derisk-v1 summary: %s", json.dumps(summary))
        if _HAS_WANDB and wandb.run is not None:
            wandb.run.summary.update({f"summary/{key}": value for key, value in summary.items()})
            for name in (
                "routing.npz",
                "probe_steps.jsonl",
                "summary.json",
                "run_metadata.json",
                "resolved_config.yaml",
            ):
                if (self.probe_dir / name).exists():
                    wandb.save(str(self.probe_dir / name), base_path=str(self.probe_dir), policy="now")


def _summarize_records(records: list[dict[str, object]], warmup_steps: int) -> dict[str, float | int]:
    """Average the per-step metrics over the steady-state steps.

    Args:
        records: Per-step metric dictionaries in step order.
        warmup_steps: Number of leading steps to exclude.

    Returns:
        ``<metric>/mean``, ``<metric>/std``, ``<metric>/min`` and ``<metric>/max`` over steps
        ``>= warmup_steps`` for the headline metrics, plus the number of steps averaged.
    """
    steady = [record for record in records if int(record["step"]) >= warmup_steps]
    summary: dict[str, float | int] = {"steps_averaged": len(steady), "warmup_steps_excluded": warmup_steps}
    for key in _SUMMARY_KEYS:
        values = np.asarray([record[key] for record in steady if record.get(key) is not None], dtype=np.float64)
        if values.size == 0:
            continue
        summary[f"{key}/mean"] = float(values.mean())
        summary[f"{key}/std"] = float(values.std())
        summary[f"{key}/min"] = float(values.min())
        summary[f"{key}/max"] = float(values.max())
    return summary


def _write_run_metadata(output_dir: Path, resolved_config: dict[str, object], probe: RoutingProbe) -> None:
    """Record what is needed to reproduce and profile this run (versions, topology, config, launch).

    Args:
        output_dir: Destination directory.
        resolved_config: The fully resolved recipe config (YAML plus CLI overrides).
        probe: The attached routing probe, for its layer / expert layout.
    """
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False))
    versions = {}
    for package in _VERSIONED_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(_ENV_PREFIXES) and not any(marker in key for marker in _SECRET_MARKERS)
    }
    metadata = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "argv": sys.argv,
        "world_size": dist.get_world_size(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
        "packages": versions,
        "environment": environment,
        "moe_layers": {"layer_ids": probe.layer_ids, "layer_names": probe.layer_names},
        "n_experts": probe.n_experts,
        "top_k": probe.top_k,
        "ep_size": probe.ep_size,
        "expert_to_ep_rank": "expert e lives on EP rank e // (n_experts // ep_size)",
        "token_categories": probe.category_names,
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,pci.bus_id,driver_version,clocks.max.sm,clocks.max.mem,power.limit",
                "--format=csv",
            ]
        ),
        "nvidia_smi_topology": _command_output(["nvidia-smi", "topo", "-m"]),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def _command_output(command: list[str]) -> str:
    """Return a command's stdout, or the error text if it cannot run."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def main() -> None:
    """Load the YAML config (plus ``--dotted.key value`` overrides) and run the probed recipe."""
    cfg = parse_args_and_load_config()
    recipe = PaddingProbeRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
