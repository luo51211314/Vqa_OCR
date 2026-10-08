#!/usr/bin/env bash
#
# OCR 模式消融研究脚本
# 测试三种数据集，三种模型，四种OCR模式（除了normal）
#

set -e

# 配置
OUTPUT_DIR="/root/autodl-tmp/codes/Vqa_ocr/ablation"
TIME="yes"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 数据集配置
DATASETS=("chartqa" "docvqa" "mydatavqa")
SPLITS=("test" "validation" "test")
METRICS=("relaxed_accuracy" "relaxed_accuracy" "weighted")

# 模型配置和对应的 conda 环境
MODELS=("llava" "qwen" "mplug3")
MODEL_TYPES=("llava" "qwen" "mplug")
MODEL_CONDA_ENVS=("vqa_infer" "vqa_train" "vqa_train")

# OCR 模式配置（不测试 normal）
OCR_MODES=("random" "LR-TB" "direct")

echo "=" * 80
echo "开始 OCR 模式消融研究"
echo "输出目录: $OUTPUT_DIR"
echo "数据集: ${DATASETS[*]}"
echo "模型: ${MODELS[*]}"
echo "OCR 模式: ${OCR_MODES[*]}"
echo "=" * 80

# 激活 conda 环境
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh

# 循环所有组合
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    SPLIT="${SPLITS[$i]}"
    METRIC="${METRICS[$i]}"
    
    for j in "${!MODELS[@]}"; do
        MODEL="${MODELS[$j]}"
        MODEL_TYPE="${MODEL_TYPES[$j]}"
        CONDA_ENV="${MODEL_CONDA_ENVS[$j]}"
        
        # 激活当前模型的 conda 环境
        conda activate "$CONDA_ENV"
        
        for OCR_MODE in "${OCR_MODES[@]}"; do
            echo ""
            echo "=" * 80
            echo "运行: $DATASET $SPLIT $MODEL $OCR_MODE (conda: $CONDA_ENV)"
            echo "=" * 80
            
            # 运行 run.sh
            bash run.sh \
                "$DATASET" \
                "$SPLIT" \
                1 \
                "" \
                "$MODEL" \
                "$MODEL_TYPE" \
                "$METRIC" \
                "manual:ocr" \
                "direct" \
                "" \
                "$TIME" \
                "no" \
                "full" \
                "$OUTPUT_DIR" \
                "no" \
                "$CONDA_ENV" \
                "$OCR_MODE"
            
            echo "完成: $DATASET $SPLIT $MODEL $OCR_MODE"
            sleep 2
        done
    done
done

echo ""
echo "=" * 80
echo "所有实验完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=" * 80
