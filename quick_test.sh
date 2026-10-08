#!/usr/bin/env bash
#
# 快速测试脚本 - 验证 OCR 模式是否正常工作
#

set -e

# 配置
OUTPUT_DIR="/root/autodl-tmp/codes/Vqa_ocr/quick_test"
CONDA_ENV="vqa_train"
TIME="yes"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=" * 80
echo "开始快速测试"
echo "输出目录: $OUTPUT_DIR"
echo "=" * 80

# 激活 conda 环境
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# ============ chartqa 数据集，三个模型，三个模式 ============

# 测试 1: chartqa + llava + random
echo ""
echo "=" * 80
echo "测试 1: chartqa + llava + random"
echo "=" * 80
bash run.sh chartqa test 1 1 llava llava relaxed_accuracy manual:ocr direct "" "$TIME" no full "$OUTPUT_DIR" no "$CONDA_ENV" random
echo "✓ 测试 1 完成"

# 测试 2: chartqa + llava + LR-TB
echo ""
echo "=" * 80
echo "测试 2: chartqa + llava + LR-TB"
echo "=" * 80
bash run.sh chartqa test 1 1 llava llava relaxed_accuracy manual:ocr direct "" "$TIME" no full "$OUTPUT_DIR" no "$CONDA_ENV" LR-TB
echo "✓ 测试 2 完成"

# 测试 3: chartqa + llava + direct
echo ""
echo "=" * 80
echo "测试 3: chartqa + llava + direct"
echo "=" * 80
bash run.sh chartqa test 1 1 llava llava relaxed_accuracy manual:ocr direct "" "$TIME" no full "$OUTPUT_DIR" no "$CONDA_ENV" direct
echo "✓ 测试 3 完成"

# 测试 4: chartqa + qwen + random
echo ""
echo "=" * 80
echo "测试 4: chartqa + qwen + random"
echo "=" * 80
bash run.sh chartqa test 1 1 qwen qwen relaxed_accuracy manual:ocr direct "" "$TIME" no full "$OUTPUT_DIR" no "$CONDA_ENV" random
echo "✓ 测试 4 完成"

# 测试 5: chartqa + mplug3 + random
echo ""
echo "=" * 80
echo "测试 5: chartqa + mplug3 + random"
echo "=" * 80
bash run.sh chartqa test 1 1 mplug3 mplug relaxed_accuracy manual:ocr direct "" "$TIME" no full "$OUTPUT_DIR" no "$CONDA_ENV" random
echo "✓ 测试 5 完成"

# ============ docvqa 数据集 ============

# 测试 6: docvqa + llava + random
echo ""
echo "=" * 80
echo "测试 6: docvqa + llava + random"
echo "=" * 80
bash run.sh docvqa validation 1 1 llava llava relaxed_accuracy manual:ocr direct "" "$TIME" no full "$OUTPUT_DIR" no "$CONDA_ENV" random
echo "✓ 测试 6 完成"

# ============ mydatavqa 数据集 ============

# 测试 7: mydatavqa + llava + random
echo ""
echo "=" * 80
echo "测试 7: mydatavqa + llava + random"
echo "=" * 80
bash run.sh mydatavqa test 1 1 llava llava weighted manual:ocr direct "" "$TIME" no full "$OUTPUT_DIR" no "$CONDA_ENV" random
echo "✓ 测试 7 完成"

# 列出生成的文件
echo ""
echo "=" * 80
echo "生成的文件:"
echo "=" * 80
ls -lh "$OUTPUT_DIR"

echo ""
echo "=" * 80
echo "快速测试完成！"
echo "=" * 80
