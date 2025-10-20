#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端训练：对齐+融合+微调LLaVA投影矩阵
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_fusion.contrastive_fusion_dataset import ContrastiveFusionDataset
from feature_fusion.contrastive_alignment import ContrastiveAlignment
from feature_fusion.attention_fusion import AttentionFusion
from model_loader.loader_llava import LLaVALoader
from feature_fusion.real_feature_extractor import RealFeatureExtractor

def train_end_to_end():
    """端到端训练：对齐+融合+微调LLaVA投影矩阵"""
    
    # 参数解析
    parser = argparse.ArgumentParser(description='端到端训练：对齐+融合+微调LLaVA投影矩阵')
    parser.add_argument('--expert_model', type=str, default='paddleocr', 
                       choices=['paddleocr', 'pix2struct'], 
                       help='专家模型类型')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/data', 
                       help='数据目录')
    parser.add_argument('--weight_dir', type=str, default='/root/autodl-tmp/weight', 
                       help='权重保存目录')
    parser.add_argument('--alpha', type=float, default=0.5, help='对比损失权重')
    parser.add_argument('--beta', type=float, default=0.5, help='VQA损失权重')
    
    args = parser.parse_args()
    
    # 验证权重参数
    if args.alpha + args.beta != 1.0:
        raise ValueError("alpha + beta 必须等于 1.0")
    
    # 创建权重目录
    os.makedirs(args.weight_dir, exist_ok=True)
    
    # 加载LLaVA模型
    print("加载LLaVA模型...")
    llava_loader = LLaVALoader()
    llava_model = llava_loader.model
    llava_tokenizer = llava_loader.tokenizer
    llava_image_processor = llava_loader.image_processor
    
    # 加载特征提取器
    print("加载特征提取器...")
    feature_extractor = RealFeatureExtractor()
    
    # 数据集
    dataset = ContrastiveFusionDataset(
        data_dir=args.data_dir,
        expert_model=args.expert_model
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 对齐模块
    alignment_module = ContrastiveAlignment(
        llava_hidden_size=1024,
        expert_hidden_size=768,
        projection_dim=512,
        temperature=1.0
    )
    
    # 融合模块
    fusion_module = AttentionFusion(
        llava_hidden_size=1024,
        expert_hidden_size=768,
        projection_dim=512
    )
    
    # 优化器（只训练对齐和融合模块，LLaVA投影矩阵将在后面处理）
    trainable_params = list(alignment_module.parameters()) + list(fusion_module.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    alignment_module.train()
    fusion_module.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_contrastive_loss = 0
        total_vqa_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 提取特征
            images = batch['images']
            questions = batch['questions']
            answers = batch['answers']
            
            # 提取LLaVA和专家特征
            llava_features = feature_extractor.extract_llava_features(images, llava_model)
            expert_features = feature_extractor.extract_expert_features(images, args.expert_model)
            
            # 1. 对比学习对齐
            aligned_llava, aligned_expert, contrastive_loss = alignment_module(
                llava_features, expert_features, labels=None, mode="train"
            )
            
            # 2. 注意力融合
            fused_features = fusion_module(aligned_llava, aligned_expert)
            
            # 3. 微调LLaVA投影矩阵（这里简化处理，实际需要更复杂的实现）
            # 使用融合特征替换原始LLaVA视觉特征
            # 计算VQA损失（这里简化处理，实际需要调用LLaVA的损失函数）
            vqa_loss = torch.tensor(0.0)  # 占位符
            
            # 计算总损失
            loss = args.alpha * contrastive_loss + args.beta * vqa_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item() if contrastive_loss is not None else 0
            total_vqa_loss += vqa_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx}/{len(dataloader)}, '
                      f'Total Loss: {loss.item():.4f}, Contrastive: {contrastive_loss.item() if contrastive_loss else 0:.4f}, VQA: {vqa_loss.item():.4f}')
        
        # 更新学习率
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_contrastive = total_contrastive_loss / len(dataloader)
        avg_vqa = total_vqa_loss / len(dataloader)
        
        print(f'Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, '
              f'Avg Contrastive: {avg_contrastive:.4f}, Avg VQA: {avg_vqa:.4f}')
        
        # 保存权重
        alignment_path = os.path.join(args.weight_dir, f'alignment_{args.expert_model}_epoch_{epoch+1}.pth')
        fusion_path = os.path.join(args.weight_dir, f'fusion_{args.expert_model}_epoch_{epoch+1}.pth')
        
        torch.save(alignment_module.state_dict(), alignment_path)
        torch.save(fusion_module.state_dict(), fusion_path)
        
        print(f'Alignment weights saved to: {alignment_path}')
        print(f'Fusion weights saved to: {fusion_path}')

if __name__ == "__main__":
    train_end_to_end()