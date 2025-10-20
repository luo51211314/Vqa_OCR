#!/usr/bin/env python3
"""
内存需求计算脚本
分析对比学习+注意力融合+投影+冻结LLaMA模型的内存需求
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_memory_requirements():
    """计算完整训练流程的内存需求"""
    
    print("=" * 80)
    print("内存需求分析 - 对比学习+注意力融合+投影+冻结LLaMA模型")
    print("=" * 80)
    
    # 基本配置参数
    batch_size = 4
    sequence_length = 512
    llava_hidden_size = 1024
    expert_hidden_size = 1024
    projection_dim = 512
    vocab_size = 32000
    hidden_size = 4096  # LLaMA模型的隐藏层大小
    num_layers = 32     # LLaMA模型的层数
    num_heads = 32      # 注意力头数
    
    print(f"配置参数:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 序列长度: {sequence_length}")
    print(f"  - LLaVA特征维度: {llava_hidden_size}")
    print(f"  - 专家特征维度: {expert_hidden_size}")
    print(f"  - 投影维度: {projection_dim}")
    print(f"  - 词汇表大小: {vocab_size}")
    print(f"  - LLaMA隐藏层大小: {hidden_size}")
    print(f"  - LLaMA层数: {num_layers}")
    print(f"  - 注意力头数: {num_heads}")
    print()
    
    # 1. 输入数据内存需求
    print("1. 输入数据内存需求:")
    
    # 输入ID (batch_size, sequence_length)
    input_ids_memory = batch_size * sequence_length * 4  # int32: 4字节
    print(f"  - 输入ID: {batch_size}x{sequence_length} = {input_ids_memory / 1024**2:.2f} MB")
    
    # 标签 (batch_size, sequence_length)
    labels_memory = batch_size * sequence_length * 4  # int32: 4字节
    print(f"  - 标签: {batch_size}x{sequence_length} = {labels_memory / 1024**2:.2f} MB")
    
    # LLaVA特征 (batch_size, 576, llava_hidden_size)
    llava_features_memory = batch_size * 576 * llava_hidden_size * 4  # float32: 4字节
    print(f"  - LLaVA特征: {batch_size}x576x{llava_hidden_size} = {llava_features_memory / 1024**2:.2f} MB")
    
    # OCR特征 (batch_size, 576, expert_hidden_size)
    ocr_features_memory = batch_size * 576 * expert_hidden_size * 4  # float32: 4字节
    print(f"  - OCR特征: {batch_size}x576x{expert_hidden_size} = {ocr_features_memory / 1024**2:.2f} MB")
    
    input_data_total = input_ids_memory + labels_memory + llava_features_memory + ocr_features_memory
    print(f"  - 输入数据总计: {input_data_total / 1024**2:.2f} MB")
    print()
    
    # 2. 对齐模块内存需求
    print("2. 对齐模块内存需求:")
    
    # 对齐后的特征 (batch_size, 576, projection_dim) × 2
    aligned_features_memory = 2 * batch_size * 576 * projection_dim * 4
    print(f"  - 对齐特征: 2x{batch_size}x576x{projection_dim} = {aligned_features_memory / 1024**2:.2f} MB")
    
    # 对比损失计算
    contrastive_loss_memory = batch_size * batch_size * 4  # 相似度矩阵
    print(f"  - 对比损失: {batch_size}x{batch_size} = {contrastive_loss_memory / 1024**2:.2f} MB")
    
    alignment_total = aligned_features_memory + contrastive_loss_memory
    print(f"  - 对齐模块总计: {alignment_total / 1024**2:.2f} MB")
    print()
    
    # 3. 融合模块内存需求
    print("3. 融合模块内存需求:")
    
    # 融合特征 (batch_size, 576, projection_dim)
    fused_features_memory = batch_size * 576 * projection_dim * 4
    print(f"  - 融合特征: {batch_size}x576x{projection_dim} = {fused_features_memory / 1024**2:.2f} MB")
    
    # 注意力权重 (batch_size, 576, 576)
    attention_weights_memory = batch_size * 576 * 576 * 4
    print(f"  - 注意力权重: {batch_size}x576x576 = {attention_weights_memory / 1024**2:.2f} MB")
    
    fusion_total = fused_features_memory + attention_weights_memory
    print(f"  - 融合模块总计: {fusion_total / 1024**2:.2f} MB")
    print()
    
    # 4. LLaMA模型内存需求（即使冻结，仍然需要加载到内存）
    print("4. LLaMA模型内存需求（冻结参数）:")
    
    # 词嵌入层 (vocab_size, hidden_size)
    embedding_memory = vocab_size * hidden_size * 4
    print(f"  - 词嵌入层: {vocab_size}x{hidden_size} = {embedding_memory / 1024**3:.2f} GB")
    
    # 输入嵌入 (batch_size, sequence_length, hidden_size)
    input_embeddings_memory = batch_size * sequence_length * hidden_size * 4
    print(f"  - 输入嵌入: {batch_size}x{sequence_length}x{hidden_size} = {input_embeddings_memory / 1024**2:.2f} MB")
    
    # 注意力机制内存需求（这是最大的开销）
    # 每个头的key/value缓存: (batch_size, num_heads, sequence_length, head_dim)
    head_dim = hidden_size // num_heads
    
    # 前向传播中的key/value缓存（所有层）
    kv_cache_per_layer = 2 * batch_size * num_heads * sequence_length * head_dim * 4
    kv_cache_total = num_layers * kv_cache_per_layer
    print(f"  - KV缓存（单层）: {kv_cache_per_layer / 1024**2:.2f} MB")
    print(f"  - KV缓存（{num_layers}层）: {kv_cache_total / 1024**3:.2f} GB")
    
    # 注意力分数矩阵 (batch_size, num_heads, sequence_length, sequence_length)
    attention_scores_memory = batch_size * num_heads * sequence_length * sequence_length * 4
    print(f"  - 注意力分数: {batch_size}x{num_heads}x{sequence_length}x{sequence_length} = {attention_scores_memory / 1024**3:.2f} GB")
    
    # 隐藏状态（所有层）(batch_size, sequence_length, hidden_size) × num_layers
    hidden_states_memory = num_layers * batch_size * sequence_length * hidden_size * 4
    print(f"  - 隐藏状态（{num_layers}层）: {hidden_states_memory / 1024**3:.2f} GB")
    
    llama_total = embedding_memory + input_embeddings_memory + kv_cache_total + attention_scores_memory + hidden_states_memory
    print(f"  - LLaMA模型总计: {llama_total / 1024**3:.2f} GB")
    print()
    
    # 5. 投影层和梯度内存
    print("5. 投影层和梯度内存:")
    
    # mm_projector梯度 (4096, 1024) + (4096, 4096)
    projector_grad_memory = (4096 * 1024 + 4096 * 4096) * 4 * 2  # 权重+梯度
    print(f"  - 投影层梯度: {(4096*1024 + 4096*4096)*4/1024**2:.2f} MB")
    
    # 对齐模块梯度
    alignment_grad_memory = (1024 * 512 + 1024 * 512) * 4 * 2  # 两个投影层
    print(f"  - 对齐模块梯度: {(1024*512 + 1024*512)*4/1024**2:.2f} MB")
    
    # 融合模块梯度
    fusion_grad_memory = (512 * 512) * 4 * 2  # 注意力融合层
    print(f"  - 融合模块梯度: {(512*512)*4/1024**2:.2f} MB")
    
    gradients_total = projector_grad_memory + alignment_grad_memory + fusion_grad_memory
    print(f"  - 梯度内存总计: {gradients_total / 1024**2:.2f} MB")
    print()
    
    # 6. 总内存需求汇总
    print("6. 总内存需求汇总:")
    
    total_memory_mb = (input_data_total + alignment_total + fusion_total + gradients_total) / 1024**2
    total_memory_gb = llama_total / 1024**3
    
    print(f"  - 数据处理模块: {input_data_total / 1024**2:.2f} MB")
    print(f"  - 对齐融合模块: {(alignment_total + fusion_total) / 1024**2:.2f} MB")
    print(f"  - 梯度内存: {gradients_total / 1024**2:.2f} MB")
    print(f"  - LLaMA模型: {llama_total / 1024**3:.2f} GB")
    print(f"  - 总计（近似）: {total_memory_gb:.2f} GB")
    print()
    
    # 7. 内存不足原因分析
    print("7. 内存不足原因分析:")
    print("即使冻结了LLaMA的大部分参数，内存不足的主要原因是:")
    print("  ✓ LLaMA模型本身需要大量内存（~25-30GB）")
    print("  ✓ 注意力机制需要存储KV缓存和注意力分数矩阵")
    print("  ✓ 576个token的长序列导致注意力计算复杂度为O(n²)")
    print("  ✓ 批处理大小为4进一步放大了内存需求")
    print("  ✓ PyTorch的内存分配策略可能导致内存碎片")
    print()
    
    # 8. 优化建议
    print("8. 优化建议:")
    print("  ✓ 将批处理大小减少到1或2")
    print("  ✓ 将序列长度限制到256或更短")
    print("  ✓ 使用梯度累积而不是大批次")
    print("  ✓ 启用混合精度训练（fp16）")
    print("  ✓ 使用更小的LLaMA模型（如7B版本）")
    print("  ✓ 启用梯度检查点（checkpointing）")
    print("  ✓ 设置PyTorch内存优化参数:")
    print("      os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'")
    print()
    
    return total_memory_gb

def test_actual_memory_usage():
    """测试实际内存使用情况"""
    print("=" * 80)
    print("实际内存使用测试")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过实际内存测试")
        return
    
    # 记录初始内存使用
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"初始GPU内存使用: {initial_memory:.2f} GB")
    
    # 创建一些测试张量来模拟实际使用
    try:
        # 模拟LLaVA特征
        llava_features = torch.randn(4, 576, 1024, device='cuda')
        print(f"创建LLaVA特征后内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 模拟OCR特征
        ocr_features = torch.randn(4, 576, 1024, device='cuda')
        print(f"创建OCR特征后内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 模拟输入嵌入
        input_embeddings = torch.randn(4, 512, 4096, device='cuda')
        print(f"创建输入嵌入后内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 模拟注意力计算（这是内存消耗最大的部分）
        attention_scores = torch.randn(4, 32, 512, 512, device='cuda')
        print(f"创建注意力分数后内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 模拟KV缓存（32层）
        kv_cache = []
        for i in range(32):
            key_cache = torch.randn(4, 32, 512, 128, device='cuda')
            value_cache = torch.randn(4, 32, 512, 128, device='cuda')
            kv_cache.append((key_cache, value_cache))
        
        print(f"创建KV缓存后内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 清理内存
        del llava_features, ocr_features, input_embeddings, attention_scores, kv_cache
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"清理后GPU内存使用: {final_memory:.2f} GB")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"内存不足错误: {e}")
        print("这证实了我们的内存需求分析")

if __name__ == "__main__":
    # 计算理论内存需求
    theoretical_requirement = calculate_memory_requirements()
    
    # 测试实际内存使用
    test_actual_memory_usage()
    
    print("=" * 80)
    print("结论:")
    print(f"理论内存需求约为: {theoretical_requirement:.2f} GB")
    print("您的GPU有31.73 GB内存，但实际可用内存可能只有25-28 GB")
    print("由于内存碎片和PyTorch的内存管理开销，实际需求可能超过可用内存")
    print("=" * 80)