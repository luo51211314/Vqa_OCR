#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理parquet数据集中的图像，进行OCR识别并保存原始结果
"""

import os
import json
import hashlib
import pandas as pd
from PIL import Image
import io
import glob
from tqdm import tqdm
import sys
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 输出目录配置
OUTPUT_DIR = "/root/autodl-tmp/dataset/myPlot"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
PROCESSED_MARKER = os.path.join(OUTPUT_DIR, "processed_images.json")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# 加载已处理的图片标记
processed_images = {}
if os.path.exists(PROCESSED_MARKER):
    with open(PROCESSED_MARKER, "r", encoding="utf-8") as f:
        processed_images = json.load(f)
    print(f"已加载 {len(processed_images)} 个已处理图片的标记")


def get_image_hash(image_bytes):
    """计算图像的MD5哈希值，用于识别重复图片"""
    return hashlib.md5(image_bytes).hexdigest()


def save_image(image_bytes, image_id):
    """保存原始图片到文件"""
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    return image_path


def load_ocr_expert():
    """加载OCR专家模块"""
    print("正在加载OCR专家模块...")
    try:
        from expert.ocr_expert import OcrExpert
        ocr_expert = OcrExpert()
        ocr_expert.initialize()
        print("OCR专家模块加载成功\n")
        return ocr_expert
    except Exception as e:
        print(f"加载OCR专家模块失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_single_image(image_bytes, image_id, ocr_expert):
    """处理单张图片：OCR识别并保存原始结果"""
    try:
        # 从字节读取图片
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size
        image_np = np.array(image)
        
        # 预处理图像：转换为RGB格式，去除alpha通道
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        
        # 使用OCR专家获取文本块
        ocr_result = ocr_expert.process(image_np)
        
        # 处理polygon字段，避免序列化问题
        if "ocr_texts" in ocr_result:
            for text_item in ocr_result["ocr_texts"]:
                if "polygon" in text_item:
                    # 移除polygon字段，或者转换为列表
                    del text_item["polygon"]
        
        # 保存JSON结果（只保存原始OCR数据）
        json_path = os.path.join(JSON_DIR, f"{image_id}.json")
        output_data = {
            "image_id": image_id,
            "image_size": [img_width, img_height],
            "ocr_raw_result": ocr_result
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return True, json_path
    except Exception as e:
        print(f"\n处理图片 {image_id} 时出错: {e}")
        return False, str(e)


def process_parquet_file(parquet_path, ocr_expert):
    """处理单个parquet文件"""
    print(f"\n处理文件: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    file_name = os.path.basename(parquet_path)
    base_name = os.path.splitext(file_name)[0]
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 使用tqdm进度条
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=file_name, unit="img"):
        image_bytes = row["image_bytes"]
        
        # 计算图片哈希，检查是否已处理
        img_hash = get_image_hash(image_bytes)
        if img_hash in processed_images:
            skip_count += 1
            continue
        
        # 生成唯一的图片ID
        image_id = f"{base_name}_{idx:04d}"
        
        # 保存原始图片
        save_image(image_bytes, image_id)
        
        # OCR识别并保存结果
        success, result = process_single_image(image_bytes, image_id, ocr_expert)
        
        if success:
            success_count += 1
            processed_images[img_hash] = {
                "image_id": image_id,
                "parquet_file": file_name,
                "index": idx
            }
        else:
            error_count += 1
    
    print(f"{file_name} 处理完成: 成功 {success_count}, 跳过 {skip_count}, 错误 {error_count}")
    return success_count, skip_count, error_count


def main():
    """主函数"""
    # 查找所有article_train_batch_*.parquet文件
    parquet_pattern = "/root/autodl-tmp/dataset/llava_pretrain_data/article_train_batch_*.parquet"
    parquet_files = sorted(glob.glob(parquet_pattern))
    
    if not parquet_files:
        print(f"未找到匹配的parquet文件: {parquet_pattern}")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件\n")
    
    # 加载OCR专家
    ocr_expert = load_ocr_expert()
    if ocr_expert is None:
        print("无法加载OCR专家，退出")
        return
    
    # 处理每个parquet文件
    total_success = 0
    total_skip = 0
    total_error = 0
    
    for parquet_file in parquet_files:
        success, skip, error = process_parquet_file(parquet_file, ocr_expert)
        total_success += success
        total_skip += skip
        total_error += error
        
        # 定期保存已处理标记
        with open(PROCESSED_MARKER, "w", encoding="utf-8") as f:
            json.dump(processed_images, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("全部处理完成!")
    print(f"总成功: {total_success}, 总跳过: {total_skip}, 总错误: {total_error}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
