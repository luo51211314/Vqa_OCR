#!/usr/bin/env bash

# No Clustering 模式批量测试脚本
# 不进行OCR行聚类，直接将原始OCR结果输入大模型

OUTPUT_DIR="/root/autodl-tmp/codes/Vqa_ocr/experiment_no_clustering"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "===== No Clustering 模式批量测试 ====="
echo "输出目录: $OUTPUT_DIR"
echo ""

# ==================== ChartQA test 集 ====================
echo ""
echo "############################################################"
echo "#              ChartQA test 集 - No Clustering              #"
echo "############################################################"
echo ""

# OCR + LLaVA
echo "=== ChartQA test - OCR + LLaVA (No Clustering) ==="
bash run.sh chartqa test 2 "" llava llava relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# OCR + Qwen
echo "=== ChartQA test - OCR + Qwen (No Clustering) ==="
bash run.sh chartqa test 1 "" qwen qwen relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# ==================== DocVQA val 集 ====================
echo ""
echo "############################################################"
echo "#              DocVQA val 集 - No Clustering                #"
echo "############################################################"
echo ""

# OCR + LLaVA
echo "=== DocVQA val - OCR + LLaVA (No Clustering) ==="
bash run.sh docvqa val 2 "" llava llava relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# OCR + Qwen
echo "=== DocVQA val - OCR + Qwen (No Clustering) ==="
bash run.sh docvqa val 1 "" qwen qwen relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# ==================== MyDataVQA test 集 ====================
echo ""
echo "############################################################"
echo "#              MyDataVQA test 集 - No Clustering            #"
echo "############################################################"
echo ""

# OCR + LLaVA
echo "=== MyDataVQA test - OCR + LLaVA (No Clustering) ==="
bash run.sh mydatavqa test 2 "" llava llava weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# OCR + Qwen
echo "=== MyDataVQA test - OCR + Qwen (No Clustering) ==="
bash run.sh mydatavqa test 1 "" qwen qwen weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

echo "===== No Clustering 模式批量测试完成 ====="
echo ""
echo "结果文件:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "无结果文件"
