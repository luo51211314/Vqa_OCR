#!/usr/bin/env bash

# 训练脚本执行器 - 按顺序执行阶段1和阶段2的训练
# 使用方法: bash run.sh [config_file]

CONFIG_FILE=${1:-"config.json"}
TRAIN_SCRIPT="main.py"

echo "========== VQA OCR 融合训练 =========="
echo "配置文件: $CONFIG_FILE"
echo "训练脚本: $TRAIN_SCRIPT"
echo "======================================"

# 检查Python和环境
echo "\n[步骤1] 检查环境..."
python --version
if ! python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"; then
    echo "错误: PyTorch环境配置不正确！"
    exit 1
fi

# 执行阶段1训练
echo "\n[步骤2] 开始执行阶段1训练..."
echo "时间: $(date)"
echo "命令: python $TRAIN_SCRIPT $CONFIG_FILE 1"

echo "\n======================================"
echo "            阶段1训练开始             "
echo "======================================"
python "$TRAIN_SCRIPT" "$CONFIG_FILE" 1
# 忽略退出码，直接继续执行阶段2训练
# PHASE1_STATUS=$?

# if [ $PHASE1_STATUS -ne 0 ]; then
#     echo "\n错误: 阶段1训练失败！退出码: $PHASE1_STATUS"
#     exit $PHASE1_STATUS
# fi

echo "\n[阶段1训练完成] 时间: $(date)"
echo "准备等待5秒后开始阶段2训练..."
sleep 5

# 执行阶段2训练
echo "\n[步骤3] 开始执行阶段2训练..."
echo "时间: $(date)"
echo "命令: python $TRAIN_SCRIPT $CONFIG_FILE 2"

echo "\n======================================"
echo "            阶段2训练开始             "
echo "======================================"
python "$TRAIN_SCRIPT" "$CONFIG_FILE" 2
# PHASE2_STATUS=$?

# if [ $PHASE2_STATUS -ne 0 ]; then
#     echo "\n错误: 阶段2训练失败！退出码: $PHASE2_STATUS"
#     exit $PHASE2_STATUS
# fi

echo "\n======================================"
echo "            训练全部完成！             "
echo "======================================"
echo "完成时间: $(date)"