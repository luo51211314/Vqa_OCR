#!/usr/bin/env bash
# 顺序消融：native(复用主表) / random / ros / coordinate
# 用法: bash experiments/ablations/run_order.sh [--run] [--only qwen] ...
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train

python "$REPO_ROOT/experiments/run_config.py" \
  "$REPO_ROOT/experiments/configs/ablation_order.yaml" "$@"
