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

"""CPU checks of ``ep_model`` against hand-computed batches and the recorded derisk-v1/v2 runs.

Run: ``source experiments/derisk-v1/env.sh && python experiments/choose-datasets-v1/test_ep_model.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import ep_model
import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent.parent


def test_replay_sampler_interleaves_ranks_and_pads_to_longest() -> None:
    """Rank r reads samples r, r + world, ...; each micro-batch is padded to its longest sequence."""
    # 2 ranks, 2 sequences per micro-batch, 4 per step -> 1 micro-batch per rank per step, 2 steps.
    lengths = np.array([10, 1, 30, 3, 5, 7, 2, 40])
    real, padded = ep_model.replay_sampler(lengths, world=2, local_batch=2, global_batch=4)
    # rank 0 reads 10, 30, 5, 2 -> micro-batches (10, 30), (5, 2); rank 1 reads 1, 3, 7, 40 -> (1, 3), (7, 40).
    np.testing.assert_array_equal(real, [[10 + 30 + 1 + 3], [5 + 2 + 7 + 40]])
    np.testing.assert_array_equal(padded, [[2 * 30 + 2 * 3], [2 * 5 + 2 * 40]])


def test_replay_sampler_rejects_uneven_batches() -> None:
    """Batch sizes that do not divide, or too few samples for one step, raise ``ValueError``."""
    for kwargs, message in (
        ({"lengths": np.ones(64, dtype=int), "world": 8, "local_batch": 3, "global_batch": 64}, "multiple"),
        ({"lengths": np.ones(10, dtype=int), "world": 8, "local_batch": 2, "global_batch": 128}, "fill one step"),
    ):
        try:
            ep_model.replay_sampler(**kwargs)
        except ValueError as error:
            assert message in str(error), error
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_replay_matches_recorded_derisk_microbatches() -> None:
    """The replayed shapes equal the micro-batch shapes derisk-v1's probe recorded on the GPUs."""
    lengths = np.array(json.loads((_EXPERIMENTS / "derisk-v1/runs/data_stats.json").read_text())["lengths"])
    real, padded = ep_model.replay_sampler(lengths, world=8, local_batch=2, global_batch=128)
    shapes = np.load(_EXPERIMENTS / "derisk-v1/runs/pad_batchmax_2pass/routing.npz")["microbatch_shapes"][:20]
    np.testing.assert_array_equal(padded, (shapes[..., 0] * shapes[..., 1]).sum(axis=1))
    np.testing.assert_array_equal(real, shapes[..., 2].sum(axis=1))


def test_calibrated_model_reproduces_derisk_tulu_batchmax() -> None:
    """The fitted model matches every calibration run's step time and Tulu's measured EP imbalance."""
    calibration = ep_model.fit_calibration(_EXPERIMENTS)
    for key, run in calibration.diagnostics.items():
        assert abs(run["step_time_fitted"] - run["step_time_measured"]) < 0.1, key
    lengths = np.array(json.loads((_EXPERIMENTS / "derisk-v1/runs/data_stats.json").read_text())["lengths"])
    real, padded = ep_model.replay_sampler(lengths, world=8, local_batch=2, global_batch=128)
    prediction = ep_model.predict(calibration, real, padded)
    # Measured in derisk-v1 / derisk-v2.1 (mean EP imbalance over dispatches and layers): 1.68 and 1.41.
    assert abs(prediction["eos"]["imbalance"] - 1.68) < 0.1, prediction
    assert abs(prediction["random"]["imbalance"] - 1.41) < 0.1, prediction
    assert prediction["random"]["step_time"] < prediction["eos"]["step_time"]


if __name__ == "__main__":
    test_replay_sampler_interleaves_ranks_and_pads_to_longest()
    test_replay_sampler_rejects_uneven_batches()
    test_replay_matches_recorded_derisk_microbatches()
    test_calibrated_model_reproduces_derisk_tulu_batchmax()
    print("all ep_model checks passed")
