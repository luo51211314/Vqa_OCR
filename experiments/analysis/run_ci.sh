#!/usr/bin/env bash
# 对一对 off/ocr detail CSV 计算配对 bootstrap CI + McNemar/Wilcoxon
# 用法:
#   bash experiments/analysis/run_ci.sh <off_detail.csv> <ocr_detail.csv> <out_ci.json>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
python "$REPO_ROOT/scripts/bootstrap_ci.py" --off "$1" --ocr "$2" --out "$3"
