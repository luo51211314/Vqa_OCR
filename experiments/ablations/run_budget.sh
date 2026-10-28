#!/usr/bin/env bash
# DocVQA/MyDataVQA budgets: LLaVA/Qwen/mPLUG. Existing cells skip by default.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
python "$REPO_ROOT/experiments/run_config.py" \
  "$REPO_ROOT/experiments/configs/ablation_budget.yaml" "$@"
