#!/usr/bin/env bash
# DocVQA/MyDataVQA: all 3 models; ChartQA: LLaVA/Qwen. Existing cells skip by default.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
python "$REPO_ROOT/experiments/run_config.py" \
  "$REPO_ROOT/experiments/configs/ablation_l2.yaml" "$@"
