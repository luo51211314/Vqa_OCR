#!/usr/bin/env bash
# 生成模板消融用的固定 dev300 子集 id（模板消融前必须先跑一次）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
python "$REPO_ROOT/scripts/make_dev_subset.py"
