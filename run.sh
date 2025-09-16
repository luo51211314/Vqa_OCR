#!/usr/bin/env bash
# Usage: bash run.sh [dataset] [split] [batch_size] [num_samples] [model_name] [model_type] [metric_type]
#        bash run.sh docvqa validation 4 50 llava llava anls
#        bash run.sh chartqa val 2 100 qwen qwen relaxed_accuracy

DATASET=${1:-"docvqa"}
SPLIT=${2:-"validation"}
BS=${3:-1}
NUM_SAMPLES=${4:-""}          # 空字符串表示「全部」
MODEL_NAME=${5:-"llava"}         # 模型名称，如 llava, qwen
MODEL_TYPE=${6:-"llava"}        # llava, qwen
METRIC_TYPE=${7:-"anls"}        # anls, relaxed_accuracy, relaxed_accuracy_80

# 函数：根据模型名称获取模型路径
get_model_path() {
    local model_name=$1
    local base_path="/root/autodl-tmp/codes/Vqa_ocr/models"
    local huggingface_path="/root/autodl-tmp/model"
    local model_path=""
    
    # 检查huggingface模型路径
    if [[ -d "$huggingface_path/${model_name}_hug" ]]; then
        model_path="$huggingface_path/${model_name}_hug"
    elif [[ -d "$huggingface_path/$model_name" ]]; then
        model_path="$huggingface_path/$model_name"
    else
        echo "错误：未找到模型 '$model_name'" >&2
        echo "检查的路径：" >&2
        echo "  - $base_path/$model_name" >&2
        echo "  - $huggingface_path/${model_name}_hug" >&2
        echo "  - $huggingface_path/$model_name" >&2
        echo "可用的模型：" >&2
        # 列出可用的模型
        echo "本地模型 ($base_path):" >&2
        for dir in "$base_path"/*/; do
            if [[ -d "$dir" ]]; then
                model_basename=$(basename "$dir")
                echo "  - $model_basename" >&2
            fi
        done
        echo "HuggingFace模型 ($huggingface_path):" >&2
        for dir in "$huggingface_path"/*/; do
            if [[ -d "$dir" ]]; then
                model_basename=$(basename "$dir")
                echo "  - $model_basename" >&2
            fi
        done
        exit 1
    fi
    
    echo "$model_path"
}

# 获取模型路径
MODEL_PATH=$(get_model_path "$MODEL_NAME")

echo "========== VQA Eval =========="
echo "Dataset   : $DATASET"
echo "Split     : $SPLIT"
echo "BatchSz   : $BS"
echo "Samples   : ${NUM_SAMPLES:-all}"
echo "Model     : $MODEL_TYPE"
echo "ModelName : $MODEL_NAME"
echo "ModelPath : $MODEL_PATH"
echo "Metric    : $METRIC_TYPE"
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
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --metric_type "$METRIC_TYPE"