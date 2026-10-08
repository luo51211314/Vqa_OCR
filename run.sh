#!/usr/bin/env bash

# Usage: bash run.sh [dataset] [split] [batch_size] [num_samples] [model_name] [model_type] [metric_type] [use_experts] [expert_mode] [fusion_weight] [time] [use_pipeline] [pipeline_mode] [output_dir] [debug_pipeline] [conda_env] [ocr_mode]
#        bash run.sh docvqa validation 4 50 llava llava anls auto direct "" no no full results no vqa_train normal
#        bash run.sh chartqa val 2 100 qwen qwen relaxed_accuracy manual:text,chart fusion "" no no full results no vqa_train normal
#        bash run.sh mydatavqa test 1 "" llava llava weighted manual:ocr direct "" yes yes full results no vqa_train normal
#        bash run.sh chartqa test 2 "" BLIP blip relaxed_accuracy manual:ocr direct "" no no full results no vqa_train normal
#        bash run.sh chartqa test 2 "" mplug3 mplug relaxed_accuracy manual:ocr direct "" no no full results no vqa_train normal

DATASET=${1:-"docvqa"}
SPLIT=${2:-"validation"}
BS=${3:-1}
NUM_SAMPLES=${4:-""}
MODEL_NAME=${5:-"llava"}
MODEL_TYPE=${6:-"llava"}
METRIC_TYPE=${7:-"anls"}
USE_EXPERTS=${8:-"off"}
EXPERT_MODE=${9:-"direct"}
FUSION_WEIGHT=${10:-"/root/autodl-tmp/weight/stage_2/epoch_2"}
TIME=${11:-"no"}
USE_PIPELINE=${12:-"no"}
PIPELINE_MODE=${13:-"full"}
OUTPUT_DIR=${14:-"/root/autodl-tmp/codes/Vqa_ocr/results"}
DEBUG_PIPELINE=${15:-"no"}
CONDA_ENV=${16:-"vqa_train"}
OCR_MODE=${17:-"normal"}

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

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

# 解析专家模块参数
parse_expert_args() {
    local use_experts=$1
    local expert_mode=$2
    local fusion_weight=$3
    local experts_arg=""
    local fusion_arg=""
    
    case "$use_experts" in
        "auto")
            # 自动选择专家模块
            experts_arg="--use_experts auto"
            ;;
        "off"|"none"|"false")
            # 禁用专家模块
            experts_arg=""
            ;;
        "manual"*)
            # 手动指定专家模块，格式: manual:text,chart
            if [[ "$use_experts" == manual:* ]]; then
                expert_list="${use_experts#manual:}"
                experts_arg="--use_experts manual --expert_names $expert_list"
            else
                echo "错误: 手动模式需要指定专家模块，格式: manual:text,chart" >&2
                exit 1
            fi
            ;;
        *)
            echo "错误: 无效的专家模块参数 '$use_experts'" >&2
            echo "可用选项: auto, off, manual:text,chart" >&2
            exit 1
            ;;
    esac
    
    # 添加特征融合参数
    case "$expert_mode" in
        "fusion")
            fusion_arg="--use_feature_fusion fusion"
            # 添加权重文件路径（如果提供）
            # 使用weight_dir参数替代fusion_weight
            fusion_arg="$fusion_arg --weight_dir ${fusion_weight:-/root/autodl-tmp/weight/stage_2/epoch_2}"
            ;;
        "direct")
            fusion_arg="--use_feature_fusion off"
            ;;
    esac
    
    echo "$experts_arg $fusion_arg"
}

# 验证专家模式参数
validate_expert_mode() {
    local mode=$1
    case "$mode" in
        "direct"|"fusion")
            return 0
            ;;
        *)
            echo "错误: 无效的专家模式参数 '$mode'" >&2
            echo "可用选项: direct, fusion" >&2
            echo "  direct: 直接上下游（原来的方案）" >&2
            echo "  fusion: 特征融合增强" >&2
            exit 1
            ;;
    esac
}

# 解析流水线参数
parse_pipeline_args() {
    local use_pipeline=$1
    local pipeline_mode=$2
    local debug_pipeline=$3
    local pipeline_arg=""
    
    if [[ "$use_pipeline" == "yes" ]]; then
        pipeline_arg="--use_pipeline --pipeline_mode $pipeline_mode"
        if [[ "$debug_pipeline" == "yes" ]]; then
            pipeline_arg="$pipeline_arg --debug_pipeline"
        fi
    fi
    
    echo "$pipeline_arg"
}

# 获取模型路径
MODEL_PATH=$(get_model_path "$MODEL_NAME")

# 验证专家模式
validate_expert_mode "$EXPERT_MODE"

# 解析专家参数
EXPERTS_ARG=$(parse_expert_args "$USE_EXPERTS" "$EXPERT_MODE" "$FUSION_WEIGHT")

# 解析流水线参数
PIPELINE_ARG=$(parse_pipeline_args "$USE_PIPELINE" "$PIPELINE_MODE" "$DEBUG_PIPELINE")

echo "========== VQA Eval =========="
echo "Dataset   : $DATASET"
echo "Split     : $SPLIT"
echo "BatchSz   : $BS"
echo "Samples   : ${NUM_SAMPLES:-all}"
echo "Model     : $MODEL_TYPE"
echo "ModelName : $MODEL_NAME"
echo "ModelPath : $MODEL_PATH"
echo "Metric    : $METRIC_TYPE"
echo "Experts   : $USE_EXPERTS"
echo "ExpertMode: $EXPERT_MODE"
if [[ -n "$FUSION_WEIGHT" ]]; then
    echo "FusionWeight: $FUSION_WEIGHT"
fi
if [[ -n "$EXPERTS_ARG" ]]; then
    echo "ExpertArgs: $EXPERTS_ARG"
fi
echo "Time      : $TIME"
echo "UsePipeline: $USE_PIPELINE"
echo "PipelineMode: $PIPELINE_MODE"
echo "OutputDir  : $OUTPUT_DIR"
echo "DebugPipeline: $DEBUG_PIPELINE"
echo "OcrMode   : $OCR_MODE"
echo "=============================="

# 把「空」或「all」转成 Python 的 None（脚本里用 --num_samples）
if [[ -z "$NUM_SAMPLES" || "$NUM_SAMPLES" == "all" ]]; then
    SAMPLE_ARG=""
else
    SAMPLE_ARG="--num_samples $NUM_SAMPLES"
fi

# 构建 ocr_mode 参数
OCR_MODE_ARG="--ocr_mode $OCR_MODE"

CUDA_VISIBLE_DEVICES=0 \
python -m main_inference \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --bs "$BS" \
  $SAMPLE_ARG \
  --model_path "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --metric_type "$METRIC_TYPE" \
  $EXPERTS_ARG \
  --time "$TIME" \
  --output_dir "$OUTPUT_DIR" \
  $PIPELINE_ARG \
  $OCR_MODE_ARG
