#!/usr/bin/env bash
# 从某批次结果目录自动生成 Table 3（csv + tex，含 bootstrap CI/McNemar）
# 用法: bash experiments/analysis/make_table3.sh results_repro/20260601_abc123/main_table [--allow-missing]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
python "$REPO_ROOT/scripts/make_table3.py" --root "$@"
