#!/usr/bin/env bash

OUTPUT_DIR="/root/autodl-tmp/codes/Vqa_ocr/experiment"

echo "===== 开始批量测试 ====="
echo ""

# ==================== ChartQA test 集 ====================
echo ""
echo "############################################################"
echo "#                    ChartQA test 集                        #"
echo "############################################################"
echo ""

# OCR + LLaVA
echo "=== ChartQA test - OCR + LLaVA ==="
bash run.sh chartqa test 2 "" llava llava relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer
echo ""

# OCR + Qwen
echo "=== ChartQA test - OCR + Qwen ==="
bash run.sh chartqa test 2 "" qwen qwen relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

# 无OCR + LLaVA
echo "=== ChartQA test - 无OCR + LLaVA ==="
bash run.sh chartqa test 2 "" llava llava relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_infer
echo ""

# 无OCR + Qwen
echo "=== ChartQA test - 无OCR + Qwen ==="
bash run.sh chartqa test 2 "" qwen qwen relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

# ==================== DocVQA val 集 ====================
echo ""
echo "############################################################"
echo "#                    DocVQA val 集                          #"
echo "############################################################"
echo ""

# OCR + LLaVA
echo "=== DocVQA val - OCR + LLaVA ==="
bash run.sh docvqa val 2 "" llava llava relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer
echo ""

# OCR + Qwen
echo "=== DocVQA val - OCR + Qwen ==="
bash run.sh docvqa val 2 "" qwen qwen relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

# 无OCR + LLaVA
echo "=== DocVQA val - 无OCR + LLaVA ==="
bash run.sh docvqa val 2 "" llava llava relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_infer
echo ""

# 无OCR + Qwen
echo "=== DocVQA val - 无OCR + Qwen ==="
bash run.sh docvqa val 2 "" qwen qwen relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

# ==================== MyDataVQA test 集 ====================
echo ""
echo "############################################################"
echo "#                    MyDataVQA test 集                      #"
echo "############################################################"
echo ""

# OCR + LLaVA
echo "=== MyDataVQA test - OCR + LLaVA ==="
bash run.sh mydatavqa test 2 "" llava llava weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer
echo ""

# OCR + Qwen
echo "=== MyDataVQA test - OCR + Qwen ==="
bash run.sh mydatavqa test 2 "" qwen qwen weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

# 无OCR + LLaVA
echo "=== MyDataVQA test - 无OCR + LLaVA ==="
bash run.sh mydatavqa test 2 "" llava llava weighted off direct "" yes no full "$OUTPUT_DIR" no vqa_infer
echo ""

# 无OCR + Qwen
echo "=== MyDataVQA test - 无OCR + Qwen ==="
bash run.sh mydatavqa test 2 "" qwen qwen weighted off direct "" yes no full "$OUTPUT_DIR" no vqa_train
echo ""

echo ""
echo "===== 所有测试完成！====="
echo ""
echo "结果保存位置: $OUTPUT_DIR"
echo ""
