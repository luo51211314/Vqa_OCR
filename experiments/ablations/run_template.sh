#!/usr/bin/env bash
# 模板消融 P1/P2/P3（llava+qwen，dev300 子集，禁止用测试集选模板）
# 前置: python scripts/make_dev_subset.py
# 用法: bash experiments/ablations/run_template.sh [--run]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train

python "$REPO_ROOT/experiments/run_config.py" \
  "$REPO_ROOT/experiments/configs/ablation_template.yaml" "$@"
