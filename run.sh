#!/usr/bin/env bash
# Usage: bash run.sh docvqa test 4 "" /root/xxx
#        bash run.sh docvqa test 4 50   # 只跑 50 条

DATASET=${1:-"docvqa"}
SPLIT=${2:-"validation"}
BS=${3:-1}
NUM_SAMPLES=${4:-""}          # 空字符串表示「全部」
MODEL_PATH=${5:-"/root/autodl-tmp/model/llava_hug"}

echo "========== VQA Eval =========="
echo "Dataset : $DATASET"
echo "Split   : $SPLIT"
echo "BatchSz : $BS"
echo "Samples : ${NUM_SAMPLES:-all}"
echo "Model   : $MODEL_PATH"
echo "=============================="

# 把「空」或「all」转成 Python 的 None（脚本里用 --num_samples）
if [[ -z "$NUM_SAMPLES" || "$NUM_SAMPLES" == "all" ]]; then
    SAMPLE_ARG=""
else
    SAMPLE_ARG="--num_samples $NUM_SAMPLES"
fi

CUDA_VISIBLE_DEVICES=0 \
python -m main_inference \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --bs "$BS" \
  $SAMPLE_ARG \
  --model_path "$MODEL_PATH"