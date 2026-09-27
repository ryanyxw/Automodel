#!/usr/bin/env bash
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

# Build the Python environment for main-v1 on this host (uv only).
#
# Why not `uv sync --locked` or the NGC container: this host runs NVIDIA driver 550.90.07 (CUDA 12.4).
# uv.lock pins torch 2.10.0+cu130 and nvcr.io/nvidia/nemo-automodel:26.08 is built on CUDA 13.3; both
# need a >=580/610 driver, and CUDA forward compatibility is unavailable on R550. This script keeps the
# uv.lock versions but swaps in PyPI's torch 2.10.0 (CUDA 12.8, which R550 runs through CUDA 12.x
# minor-version compatibility) and compiles the CUDA extensions that have no matching prebuilt wheel
# for sm_90 with the host nvcc 12.4. pyproject.toml and uv.lock are not modified; the checkout is used
# through PYTHONPATH (see env.sh).
#
# Usage: bash experiments/main-v1/setup_env.sh [base] [builds]   (default: both stages)
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${EXP_VENV:-$HOME/venvs/automodel-cu12}
UV=${UV:-$HOME/.local/bin/uv}
LOG_DIR=${EXP_ENV_LOG_DIR:-$HOME/venvs/logs}
mkdir -p "$LOG_DIR"
export UV_HTTP_TIMEOUT=300

stage_base() {
  "$UV" venv "$VENV" --python 3.12 --seed
  # Locked base dependencies, minus torch and the CUDA-13 wheels it pulls in (and docs/test tools).
  (cd "$REPO" && "$UV" export --locked --no-hashes --no-emit-project --no-dev --format requirements-txt) \
    | grep -v "^\s*#" | grep -v "^\s*$" \
    | grep -vE "^(torch==|triton==|cuda-(bindings|python|pathfinder)==|nvidia-(cublas|cuda-|cudnn|cufft|cufile|curand|cusolver|cusparse|nccl|nvjitlink|nvshmem|nvtx|sphinx)|onnxruntime|sphinx|pytest|coverage|astroid|myst-parser|pydata-sphinx|alabaster|accessible-pygments|docutils|snowballstemmer|imagesize|roman-numerals|mdit-py-plugins)" \
    > "$VENV/base-reqs.txt"
  (cd "$HOME" && "$UV" pip install --python "$VENV/bin/python" -r "$VENV/base-reqs.txt" \
    "torch==2.10.0" "triton==3.6.0" wandb "onnxscript>=0.5.6" ninja wheel)
  # FusedLinearCrossEntropy backend (pure Python + Triton), at the [tool.uv.sources] revision.
  (cd "$HOME" && "$UV" pip install --python "$VENV/bin/python" --no-deps \
    "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@87a86aba72cfd2f0d8abecaf81c13c4528ea07d8")
}

stage_builds() {
  local site_packages status=0
  site_packages=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  # These standalone installs must not pick up the checkout's [tool.uv] build settings.
  export UV_NO_CONFIG=1
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
  # NCCL / cuDNN headers come from the nvidia-*-cu12 wheels that torch installed.
  export CPATH="$site_packages/nvidia/nccl/include:$site_packages/nvidia/cudnn/include${CPATH:+:$CPATH}"
  export TORCH_CUDA_ARCH_LIST="9.0"
  export NVTE_CUDA_ARCHS="90"
  export NVTE_FRAMEWORK=pytorch
  # TE's own NCCL expert-parallel extension needs torch symmetric-memory headers that the CUDA 12.8
  # torch wheel does not ship. AutoModel dispatches experts with DeepEP, so it is not needed here.
  export NVTE_WITH_NCCL_EP=0
  export MAX_JOBS=64

  # cuDNN 9.18 instead of torch's 9.10 pin (pyproject.toml recommends >=9.18 for TE fused attention).
  "$UV" pip install --python "$VENV/bin/python" --no-deps "nvidia-cudnn-cu12==9.18.1.3" \
    > "$LOG_DIR/cudnn.log" 2>&1 || { echo "cuDNN install failed (see $LOG_DIR/cudnn.log)"; status=1; }
  # TransformerEngine: the CUDA-12 core wheel plus the PyTorch extension compiled against this torch.
  # --no-deps keeps transformer-engine-torch from also pulling the CUDA-13 core (transformer-engine-cu13).
  "$UV" pip install --python "$VENV/bin/python" --no-deps "transformer-engine==2.19.0" "transformer-engine-cu12==2.19.0" \
    > "$LOG_DIR/te_core.log" 2>&1 || { echo "TE core install failed (see $LOG_DIR/te_core.log)"; status=1; }
  "$UV" pip install --python "$VENV/bin/python" --no-deps --no-build-isolation "transformer-engine-torch==2.19.0" \
    > "$LOG_DIR/te_torch.log" 2>&1 || { echo "TE torch extension build failed (see $LOG_DIR/te_torch.log)"; status=1; }
  # Mamba kernels (compiled: no prebuilt wheel matches these versions with torch 2.10 + CUDA 12).
  "$UV" pip install --python "$VENV/bin/python" --no-build-isolation "causal-conv1d==1.6.0" \
    > "$LOG_DIR/causal_conv1d.log" 2>&1 || { echo "causal-conv1d build failed (see $LOG_DIR/causal_conv1d.log)"; status=1; }
  "$UV" pip install --python "$VENV/bin/python" --no-build-isolation "mamba-ssm==2.3.0" \
    > "$LOG_DIR/mamba_ssm.log" 2>&1 || { echo "mamba-ssm build failed (see $LOG_DIR/mamba_ssm.log)"; status=1; }
  build_deep_ep > "$LOG_DIR/deep_ep.log" 2>&1 || { echo "DeepEP build failed (see $LOG_DIR/deep_ep.log)"; status=1; }
  return $status
}

build_deep_ep() {
  # DeepEP at the uv.lock revision, built with only the classic intranode Buffer, which is all that
  # AutoModel's `dispatcher: deepep` uses on a single node. patches/deepep-intranode-only.patch adds
  # two switches: DEEPEP_DISABLE_NVSHMEM=1 takes DeepEP's existing -DDISABLE_NVSHMEM path (otherwise it
  # builds the internode kernels whenever torch's nvidia-nvshmem wheel is importable, and those need
  # InfiniBand headers this host lacks), and DEEPEP_DISABLE_HYBRID_EP=1 skips the HybridEP extension
  # (it needs CCCL and libnvtx3interop from a newer CUDA toolkit than this host's 12.4). It also makes
  # Config.get_rdma_buffer_size_hint return 0 in NVSHMEM-free builds instead of asserting: AutoModel
  # infers NVSHMEM support from Buffer.is_sm90_compiled(), which is true for this SM90 build, and so
  # asks for an RDMA buffer size that a single-node run never uses.
  local src="$HOME/venvs/src/DeepEP"
  local rev=10d4dd7377d5bce900fbb4b80cce863764892b95
  if [ ! -d "$src/.git" ]; then
    git clone https://github.com/deepseek-ai/DeepEP.git "$src" || return 1
  fi
  git -C "$src" fetch --quiet origin "$rev" && git -C "$src" checkout --quiet --force "$rev" || return 1
  git -C "$src" apply "$REPO/experiments/main-v1/patches/deepep-intranode-only.patch" || return 1
  rm -f "$HOME"/venvs/wheels/deep_ep-*.whl
  DEEPEP_DISABLE_NVSHMEM=1 DEEPEP_DISABLE_HYBRID_EP=1 "$UV" build --wheel --no-build-isolation \
    --python "$VENV/bin/python" --out-dir "$HOME/venvs/wheels" "$src" || return 1
  # --no-deps keeps uv from re-resolving torch; DeepEP's one runtime dependency is pynvml.
  "$UV" pip install --python "$VENV/bin/python" --no-deps "$HOME"/venvs/wheels/deep_ep-*.whl nvidia-ml-py
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(base builds)
rc=0
for stage in "${stages[@]}"; do
  echo "=== stage: $stage ($(date -u +%H:%M:%S))"
  "stage_$stage" || rc=1
done
echo "=== done ($(date -u +%H:%M:%S)), exit $rc"
exit $rc
