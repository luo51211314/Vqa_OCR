#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实特征提取器
"""

import sys
import os
sys.path.append('/root/autodl-tmp/codes/Vqa_ocr')

from real_feature_extractor import RealFeatureExtractor
import torch

def test_real_feature_extractor():
    """测试真实特征提取器"""
    print("=== 测试真实特征提取器 ===")
    
    try:
        # 测试LLaVA特征提取器
        print("\n1. 测试LLaVA特征提取器...")
        llava_extractor = RealFeatureExtractor(model_type="llava")
        
        # 创建一个测试图像（这里使用随机图像路径，实际使用时需要真实图像）
        test_image_path = "/tmp/test_image.png"
        
        # 如果测试图像不存在，创建一个简单的测试图像
        from PIL import Image
        import numpy as np
        
        if not os.path.exists(test_image_path):
            print("创建测试图像...")
            # 创建一个简单的测试图像
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_image = Image.fromarray(img_array)
            test_image.save(test_image_path)
        
        # 提取LLaVA特征
        llava_features = llava_extractor.extract_features(
            test_image_path, "What is in this image?")
        print(f"LLaVA特征形状: {llava_features.shape}")
        print(f"LLaVA特征范围: [{llava_features.min():.4f}, {llava_features.max():.4f}]")
        
        # 测试PaddleOCR特征提取器
        print("\n2. 测试PaddleOCR特征提取器...")
        paddleocr_extractor = RealFeatureExtractor(model_type="paddleocr")
        
        # 提取PaddleOCR特征
        paddleocr_features = paddleocr_extractor.extract_features(test_image_path)
        print(f"PaddleOCR特征形状: {paddleocr_features.shape}")
        print(f"PaddleOCR特征范围: [{paddleocr_features.min():.4f}, {paddleocr_features.max():.4f}]")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_dataset_with_real_extractor():
    """测试使用真实特征提取器的数据集"""
    print("\n=== 测试数据集与真实特征提取器 ===")
    
    try:
        from feature_fusion_dataset import ContrastiveFusionDataset
        
        # 创建数据集
        dataset = ContrastiveFusionDataset(
            dataset_path="/root/autodl-tmp/dataset/trainingVQA",
            expert_model="paddleocr",
            llava_model_path="/root/autodl-tmp/model/llava_hug"
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"样本键: {list(sample.keys())}")
        print(f"LLaVA特征形状: {sample['llava_features'].shape}")
        print(f"专家特征形状: {sample['expert_features'].shape}")
        print(f"标签: {sample['labels']}")
        print(f"图像ID: {sample['image_id']}")
        
        print("\n=== 数据集测试完成 ===")
        
    except Exception as e:
        print(f"数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_feature_extractor()
    test_dataset_with_real_extractor()