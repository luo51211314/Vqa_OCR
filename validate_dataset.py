import pandas as pd
import os
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def validate_trainingVQA_dataset():
    """验证trainingVQA数据集是否符合要求"""
    
    dataset_path = "/root/autodl-tmp/dataset/trainingVQA"
    
    print("=== 验证trainingVQA数据集 ===")
    print(f"数据集路径: {dataset_path}")
    
    # 1. 检查文件是否存在
    train_file = os.path.join(dataset_path, "train_contrastive_fusion_train.parquet")
    val_file = os.path.join(dataset_path, "val_contrastive_fusion_validation.parquet")
    
    print(f"\n1. 文件检查:")
    print(f"训练集文件: {train_file} - {'存在' if os.path.exists(train_file) else '不存在'}")
    print(f"验证集文件: {val_file} - {'存在' if os.path.exists(val_file) else '不存在'}")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("❌ 文件不存在，验证失败")
        return False
    
    # 2. 读取并检查文件结构
    print(f"\n2. 文件结构检查:")
    
    try:
        train_df = pd.read_parquet(train_file)
        val_df = pd.read_parquet(val_file)
        
        print(f"训练集样本数: {len(train_df)}")
        print(f"验证集样本数: {len(val_df)}")
        print(f"训练集列名: {list(train_df.columns)}")
        print(f"验证集列名: {list(val_df.columns)}")
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    # 3. 检查列结构是否符合要求
    print(f"\n3. 列结构检查:")
    
    # 实际数据集列结构
    actual_columns = ['question', 'answer', 'source', 'ocr_text', 'questionId', 'image_bytes']
    train_columns = list(train_df.columns)
    val_columns = list(val_df.columns)
    
    print(f"实际数据集列结构: {actual_columns}")
    print(f"训练集实际列: {train_columns}")
    print(f"验证集实际列: {val_columns}")
    
    # 检查列是否匹配实际结构
    columns_match = (set(train_columns) == set(actual_columns) and 
                    set(val_columns) == set(actual_columns))
    
    if columns_match:
        print("✅ 列结构符合实际数据集要求")
    else:
        print("❌ 列结构不符合实际数据集要求")
        return False
    
    # 4. 检查数据类型
    print(f"\n4. 数据类型检查:")
    
    print(f"训练集数据类型:")
    for col in train_df.columns:
        print(f"  {col}: {train_df[col].dtype}")
    
    print(f"验证集数据类型:")
    for col in val_df.columns:
        print(f"  {col}: {val_df[col].dtype}")
    
    # 5. 检查数据分布（数据来源比例）
    print(f"\n5. 数据分布检查:")
    
    print(f"训练集数据来源分布:")
    train_source_counts = train_df['source'].value_counts()
    print(train_source_counts)
    
    print(f"验证集数据来源分布:")
    val_source_counts = val_df['source'].value_counts()
    print(val_source_counts)
    
    # 6. 检查图像数据完整性
    print(f"\n6. 图像数据完整性检查:")
    
    # 检查训练集图像
    train_image_check = []
    for i, row in train_df.head(5).iterrows():
        try:
            if 'image_bytes' in row and row['image_bytes'] is not None:
                image = Image.open(io.BytesIO(row['image_bytes']))
                train_image_check.append(f"样本 {i}: {image.size} - ✅ 有效")
            else:
                train_image_check.append(f"样本 {i}: ❌ 无图像数据")
        except Exception as e:
            train_image_check.append(f"样本 {i}: ❌ 图像读取失败 - {e}")
    
    print("训练集图像检查:")
    for check in train_image_check:
        print(f"  {check}")
    
    # 检查验证集图像
    val_image_check = []
    for i, row in val_df.head(5).iterrows():
        try:
            if 'image_bytes' in row and row['image_bytes'] is not None:
                image = Image.open(io.BytesIO(row['image_bytes']))
                val_image_check.append(f"样本 {i}: {image.size} - ✅ 有效")
            else:
                val_image_check.append(f"样本 {i}: ❌ 无图像数据")
        except Exception as e:
            val_image_check.append(f"样本 {i}: ❌ 图像读取失败 - {e}")
    
    print("验证集图像检查:")
    for check in val_image_check:
        print(f"  {check}")
    
    # 7. 随机样本检查
    print(f"\n7. 随机样本检查:")
    
    # 训练集随机样本
    print("训练集随机样本:")
    random_train_samples = train_df.sample(min(15, len(train_df)))
    for idx, row in random_train_samples.iterrows():
        print(f"样本 {idx}:")
        print(f"  问题: {row['question'][:100] if row['question'] else '空'}")
        print(f"  答案: {row['answer'][:100] if row['answer'] else '空'}")
        print(f"  数据来源: {row['source']}")
        print(f"  OCR文本: {row['ocr_text'][:100] if row['ocr_text'] else '空'}")
        print(f"  问题ID: {row['questionId']}")
        print()
    
    # 验证集随机样本
    print("验证集随机样本:")
    random_val_samples = val_df.sample(min(5, len(val_df)))
    for idx, row in random_val_samples.iterrows():
        print(f"样本 {idx}:")
        print(f"  问题: {row['question'][:100] if row['question'] else '空'}")
        print(f"  答案: {row['answer'][:100] if row['answer'] else '空'}")
        print(f"  数据来源: {row['source']}")
        print(f"  OCR文本: {row['ocr_text'][:100] if row['ocr_text'] else '空'}")
        print(f"  问题ID: {row['questionId']}")
        print()
    
    # 8. 统计信息
    print(f"\n8. 统计信息:")
    
    print(f"训练集:")
    print(f"  总样本数: {len(train_df)}")
    print(f"  数据来源种类: {train_df['source'].nunique()}")
    print(f"  问题平均长度: {train_df['question'].str.len().mean():.2f}")
    print(f"  答案平均长度: {train_df['answer'].str.len().mean():.2f}")
    print(f"  OCR文本平均长度: {train_df['ocr_text'].str.len().mean():.2f}")
    
    print(f"验证集:")
    print(f"  总样本数: {len(val_df)}")
    print(f"  数据来源种类: {val_df['source'].nunique()}")
    print(f"  问题平均长度: {val_df['question'].str.len().mean():.2f}")
    print(f"  答案平均长度: {val_df['answer'].str.len().mean():.2f}")
    print(f"  OCR文本平均长度: {val_df['ocr_text'].str.len().mean():.2f}")
    
    # 9. 验证结果总结
    print(f"\n=== 验证结果总结 ===")
    
    # 检查文件名格式
    train_filename_correct = "train_contrastive_fusion_train.parquet" in os.listdir(dataset_path)
    val_filename_correct = "val_contrastive_fusion_validation.parquet" in os.listdir(dataset_path)
    
    print(f"文件名格式检查:")
    print(f"  训练集以'train_'开头: {train_filename_correct}")
    print(f"  验证集以'val_'开头: {val_filename_correct}")
    
    # 检查数据完整性
    data_integrity = (len(train_df) > 0 and len(val_df) > 0 and 
                     columns_match and 
                     all(check.endswith('✅ 有效') for check in train_image_check[:3]) and
                     all(check.endswith('✅ 有效') for check in val_image_check[:3]))
    
    if data_integrity and train_filename_correct and val_filename_correct:
        print("✅ 数据集验证通过！")
        print("✅ 文件名格式正确")
        print("✅ 数据完整性良好")
        print("✅ 符合前面所述的要求")
        return True
    else:
        print("❌ 数据集验证失败")
        if not train_filename_correct:
            print("❌ 训练集文件名格式不正确")
        if not val_filename_correct:
            print("❌ 验证集文件名格式不正确")
        if not data_integrity:
            print("❌ 数据完整性有问题")
        return False

if __name__ == "__main__":
    validate_trainingVQA_dataset()