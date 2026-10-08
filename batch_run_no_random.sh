#!/usr/bin/env bash

# 无随机性 + No Clustering 模式批量测试脚本
# temperature=0, top_p=1.0, do_sample=False
# 不进行OCR行聚类，直接将原始OCR结果输入大模型

OUTPUT_DIR="/root/autodl-tmp/codes/Vqa_ocr/experiments_no_random"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "===== 无随机性 + No Clustering 模式批量测试 ====="
echo "输出目录: $OUTPUT_DIR"
echo "配置: temperature=0, top_p=1.0, do_sample=False"
echo ""

# ==================== ChartQA test 集 ====================
echo ""
echo "############################################################"
echo "#       ChartQA test 集 - No Random + No Clustering        #"
echo "############################################################"
echo ""

# OCR + mPLUG3
echo "=== ChartQA test - OCR + mPLUG3 (No Random + Time) ==="
bash run.sh chartqa test 1 "" mplug3 mplug relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# 无OCR + mPLUG3
echo "=== ChartQA test - 无OCR + mPLUG3 (No Random + Time) ==="
bash run.sh chartqa test 1 "" mplug3 mplug relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# OCR + LLaVA
echo "=== ChartQA test - OCR + LLaVA (No Random + No Clustering) ==="
bash run.sh chartqa test 2 "" llava llava relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# 无OCR + LLaVA
echo "=== ChartQA test - 无OCR + LLaVA (No Random) ==="
bash run.sh chartqa test 2 "" llava llava relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# OCR + Qwen
echo "=== ChartQA test - OCR + Qwen (No Random + No Clustering) ==="
bash run.sh chartqa test 1 "" qwen qwen relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# 无OCR + Qwen
echo "=== ChartQA test - 无OCR + Qwen (No Random) ==="
bash run.sh chartqa test 1 "" qwen qwen relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# ==================== DocVQA val 集 ====================
echo ""
echo "############################################################"
echo "#       DocVQA val 集 - No Random + No Clustering          #"
echo "############################################################"
echo ""

# OCR + mPLUG3
echo "=== DocVQA val - OCR + mPLUG3 (No Random + Time) ==="
bash run.sh docvqa val 1 "" mplug3 mplug relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# 无OCR + mPLUG3
echo "=== DocVQA val - 无OCR + mPLUG3 (No Random + Time) ==="
bash run.sh docvqa val 1 "" mplug3 mplug relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# OCR + LLaVA
echo "=== DocVQA val - OCR + LLaVA (No Random + No Clustering) ==="
bash run.sh docvqa val 2 "" llava llava relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# 无OCR + LLaVA
echo "=== DocVQA val - 无OCR + LLaVA (No Random) ==="
bash run.sh docvqa val 2 "" llava llava relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# OCR + Qwen
echo "=== DocVQA val - OCR + Qwen (No Random + No Clustering) ==="
bash run.sh docvqa val 1 "" qwen qwen relaxed_accuracy manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# 无OCR + Qwen
echo "=== DocVQA val - 无OCR + Qwen (No Random) ==="
bash run.sh docvqa val 1 "" qwen qwen relaxed_accuracy off direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# ==================== MyDataVQA test 集 ====================
echo ""
echo "############################################################"
echo "#       MyDataVQA test 集 - No Random + No Clustering      #"
echo "############################################################"
echo ""

# OCR + mPLUG3
echo "=== MyDataVQA test - OCR + mPLUG3 (No Random + Time) ==="
bash run.sh mydatavqa test 1 "" mplug3 mplug weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# 无OCR + mPLUG3
echo "=== MyDataVQA test - 无OCR + mPLUG3 (No Random + Time) ==="
bash run.sh mydatavqa test 1 "" mplug3 mplug weighted off direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# OCR + LLaVA
echo "=== MyDataVQA test - OCR + LLaVA (No Random + No Clustering) ==="
bash run.sh mydatavqa test 2 "" llava llava weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# 无OCR + LLaVA
echo "=== MyDataVQA test - 无OCR + LLaVA (No Random) ==="
bash run.sh mydatavqa test 2 "" llava llava weighted off direct "" yes no full "$OUTPUT_DIR" no vqa_infer yes
echo ""

# OCR + Qwen
echo "=== MyDataVQA test - OCR + Qwen (No Random + No Clustering) ==="
bash run.sh mydatavqa test 1 "" qwen qwen weighted manual:ocr direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

# 无OCR + Qwen
echo "=== MyDataVQA test - 无OCR + Qwen (No Random) ==="
bash run.sh mydatavqa test 1 "" qwen qwen weighted off direct "" yes no full "$OUTPUT_DIR" no vqa_train yes
echo ""

echo "===== 无随机性 + No Clustering 模式批量测试完成 ====="
echo ""
echo "结果文件:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "无结果文件"
