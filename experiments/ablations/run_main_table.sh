#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
python "$REPO_ROOT/experiments/run_config.py" \
  "$REPO_ROOT/experiments/configs/main_table.yaml" "$@"
