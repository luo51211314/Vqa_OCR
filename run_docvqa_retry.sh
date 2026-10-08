#!/usr/bin/env bash

# DocVQA + MyDataVQA 重试脚本 - 运行 Qwen OCR和无OCR推理
# 修复了 prompt suffix 问题

OUTPUT_DIR="/root/autodl-tmp/codes/Vqa_ocr/experiment"

echo "===== DocVQA + MyDataVQA Qwen 重试推理 ====="
echo "输出目录: $OUTPUT_DIR"
echo ""

# ==================== DocVQA val 集 - Qwen ====================
echo ""
echo "############################################################"
echo "#                 DocVQA val - Qwen 无OCR                   #"
echo "############################################################"
echo ""

# 无OCR + Qwen
echo "=== DocVQA val - 无OCR + Qwen ==="
bash run.sh docvqa val 1 "" qwen qwen relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

echo ""
echo "############################################################"
echo "#                 DocVQA val - Qwen OCR                      #"
echo "############################################################"
echo ""

# OCR + Qwen
echo "=== DocVQA val - OCR + Qwen ==="
bash run.sh docvqa val 1 "" qwen qwen relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

# ==================== MyDataVQA test 集 - Qwen ====================
echo ""
echo "############################################################"
echo "#                 MyDataVQA test - Qwen OCR                 #"
echo "############################################################"
echo ""

# OCR + Qwen
echo "=== MyDataVQA test - OCR + Qwen ==="
bash run.sh mydatavqa test 1 "" qwen qwen weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

echo ""
echo "############################################################"
echo "#                 MyDataVQA test - Qwen 无OCR               #"
echo "############################################################"
echo ""

# 无OCR + Qwen
echo "=== MyDataVQA test - 无OCR + Qwen ==="
bash run.sh mydatavqa test 1 "" qwen qwen weighted off direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

echo "===== DocVQA + MyDataVQA Qwen 重试推理完成 ====="
echo ""
echo "结果文件:"
ls -lh "$OUTPUT_DIR"/docvqa_val_qwen_*.csv "$OUTPUT_DIR"/mydatavqa_test_qwen_*.csv 2>/dev/null || echo "无结果文件"
