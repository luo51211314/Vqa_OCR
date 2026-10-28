#!/usr/bin/env bash
# 主表 12 格（ChartQA test 2500 + DocVQA val 5349 × 3 模型 × off/ocr）
# 用法:
#   bash experiments/main_table/run_main_table.sh            # dry-run 预览
#   bash experiments/main_table/run_main_table.sh --run      # 真正执行
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train

python "$REPO_ROOT/experiments/run_config.py" \
  "$REPO_ROOT/experiments/configs/main_table.yaml" "$@"
