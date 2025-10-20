import torch
import torch.nn as nn
import argparse
import os
import sys
import pandas as pd
from PIL import Image
import json
import io
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_fusion.contrastive_alignment import ContrastiveAlignment
from feature_fusion.attention_fusion import AttentionFusion
from feature_fusion.real_feature_extractor import RealFeatureExtractor
from model_loader.loader_llava import LLaVALoader

def analyze_predictions():
    """分析预测结果以查找问题"""
    
    # 参数解析
    parser = argparse.ArgumentParser(description='分析预测结果')
    parser.add_argument('--expert_model', type=str, default='paddleocr', 
                       choices=['paddleocr', 'pix2struct'], 
                       help='专家模型类型')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/dataset/trainingVQA', 
                       help='数据目录')
    parser.add_argument('--weight_dir', type=str, default='/root/autodl-tmp/weight', 
                       help='权重保存目录')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='训练设备')
    parser.add_argument('--sample_count', type=int, default=10, help='分析样本数量')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载LLaVA模型
    print("加载LLaVA模型...")
    llava_loader = LLaVALoader()
    model_path = "/root/autodl-tmp/model/llava_hug"
    llava_tokenizer, llava_model, llava_image_processor, context_len = llava_loader.load_model(model_path, device=args.device)
    
    # 加载权重
    alignment_module = ContrastiveAlignment(
        llava_hidden_size=1024,
        expert_hidden_size=1024,
        projection_dim=512,
        temperature=1.0
    ).to(device)
    
    fusion_module = AttentionFusion(
        llava_hidden_size=1024,
        expert_hidden_size=1024,
        projection_dim=512
    ).to(device)
    
    # 加载最新权重
    alignment_weights = None
    fusion_weights = None
    llava_proj_weights = None
    
    # 查找最新的权重文件
    weight_files = os.listdir(args.weight_dir)
    alignment_files = [f for f in weight_files if f.startswith('alignment') and f.endswith('.pth')]
    fusion_files = [f for f in weight_files if f.startswith('fusion') and f.endswith('.pth')]
    llava_proj_files = [f for f in weight_files if f.startswith('llava_projection') and f.endswith('.pth')]
    
    if alignment_files:
        latest_alignment = sorted(alignment_files)[-1]
        alignment_weights = torch.load(os.path.join(args.weight_dir, latest_alignment), map_location=device)
        alignment_module.load_state_dict(alignment_weights)
        print(f"加载对齐模块权重: {latest_alignment}")
    
    if fusion_files:
        latest_fusion = sorted(fusion_files)[-1]
        fusion_weights = torch.load(os.path.join(args.weight_dir, latest_fusion), map_location=device)
        fusion_module.load_state_dict(fusion_weights)
        print(f"加载融合模块权重: {latest_fusion}")
    
    if llava_proj_files:
        latest_llava_proj = sorted(llava_proj_files)[-1]
        llava_proj_weights = torch.load(os.path.join(args.weight_dir, latest_llava_proj), map_location='cpu')
        # 加载到LLaVA模型
        llava_model.load_state_dict(llava_proj_weights, strict=False)
        print(f"加载LLaVA投影权重: {latest_llava_proj}")
    
    # 加载数据集（仅加载部分样本用于分析）
    print("加载数据集...")
    data_path = os.path.join(args.data_dir, 'train_contrastive_fusion_train.parquet')
    if data_path.endswith('.parquet'):
        full_data = pd.read_parquet(data_path)
        # 只取前sample_count个样本
        data = full_data.head(args.sample_count)
    else:
        raise ValueError("不支持的数据格式")
    
    # 特征提取器
    feature_extractor = RealFeatureExtractor(device=device)
    
    # 分析结果存储
    analysis_results = []
    
    print("开始分析预测结果...")
    for idx in tqdm(range(min(args.sample_count, len(data)))):
        item = data.iloc[idx]
        
        # 获取图像数据
        if 'image_bytes' in item:
            image_bytes = item['image_bytes']
            if isinstance(image_bytes, bytes):
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                image_path = os.path.join(args.data_dir, item.get('image', ''))
                image = Image.open(image_path).convert('RGB')
        else:
            image_path = os.path.join(args.data_dir, item['image'])
            image = Image.open(image_path).convert('RGB')
        
        # 获取问题和答案
        question = item['question']
        answer = item['answer']
        
        # 处理答案格式
        if hasattr(answer, '__iter__') and not isinstance(answer, str):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        answer = str(answer) if answer is not None else ""
        
        # 处理图像
        processed_image = llava_image_processor(image, return_tensors='pt')['pixel_values'].squeeze(0).to(device)
        
        # 构建输入文本
        input_text = f"<image>\n{question}\n"
        label_text = answer
        
        # Tokenize
        input_ids = llava_tokenizer.encode(input_text, max_length=args.max_length, 
                                         truncation=True, padding='max_length')
        label_ids = llava_tokenizer.encode(label_text, max_length=args.max_length, 
                                         truncation=True, padding='max_length')
        
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        label_ids = torch.tensor(label_ids, dtype=torch.long).to(device)
        
        # 提取特征
        with torch.no_grad():
            llava_features = feature_extractor.extract_llava_features(
                processed_image.unsqueeze(0), None
            ).to(device)
            
            ocr_features = feature_extractor.extract_features(
                processed_image.unsqueeze(0), 'paddleocr'
            ).to(device)
            
            # 对齐特征
            aligned_llava, aligned_expert, contrastive_loss = alignment_module(
                llava_features.squeeze(0), ocr_features.squeeze(0), labels=label_ids, mode="train"
            )
            
            # 融合特征
            fused_features = fusion_module(aligned_llava, aligned_expert)
            
            # 计算VQA损失
            vqa_loss = compute_vqa_loss(llava_model, input_ids, label_ids, fused_features, processed_image)
            
            # 生成预测
            prediction = generate_prediction(llava_model, llava_tokenizer, input_ids, fused_features, processed_image)
        
        # 存储分析结果
        analysis_results.append({
            'index': idx,
            'question': question,
            'ground_truth': answer,
            'prediction': prediction,
            'contrastive_loss': contrastive_loss.item() if contrastive_loss is not None else 0,
            'vqa_loss': vqa_loss.item(),
            'llava_feature_shape': tuple(llava_features.shape),
            'ocr_feature_shape': tuple(ocr_features.shape),
            'aligned_llava_shape': tuple(aligned_llava.shape),
            'aligned_expert_shape': tuple(aligned_expert.shape),
            'fused_feature_shape': tuple(fused_features.shape)
        })
        
        # 打印部分结果
        if idx < 3:  # 只打印前3个样本的详细信息
            print(f"\n样本 {idx+1}:")
            print(f"  问题: {question}")
            print(f"  真实答案: {answer}")
            print(f"  预测答案: {prediction}")
            print(f"  对比损失: {contrastive_loss.item() if contrastive_loss is not None else 0:.4f}")
            print(f"  VQA损失: {vqa_loss.item():.4f}")
    
    # 保存分析结果
    analysis_df = pd.DataFrame(analysis_results)
    analysis_path = os.path.join(args.weight_dir, 'prediction_analysis.csv')
    analysis_df.to_csv(analysis_path, index=False)
    print(f"分析结果已保存到: {analysis_path}")
    
    # 计算统计信息
    avg_contrastive_loss = analysis_df['contrastive_loss'].mean()
    avg_vqa_loss = analysis_df['vqa_loss'].mean()
    print(f"\n平均对比损失: {avg_contrastive_loss:.4f}")
    print(f"平均VQA损失: {avg_vqa_loss:.4f}")
    
    return analysis_results

def compute_vqa_loss(model, input_ids, labels, fused_features, image):
    """计算VQA损失"""
    
    # 准备输入
    inputs_embeds = model.model.get_input_embeddings()(input_ids)
    
    # 替换图像token的嵌入为融合特征
    from models.llava.llava.constants import IMAGE_TOKEN_INDEX
    image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
    
    if image_token_mask.any():
        image_token_positions = torch.nonzero(image_token_mask, as_tuple=True)[1]
        
        if hasattr(model, 'mm_projector'):
            projected_fused_features = model.mm_projector(fused_features)
        else:
            projected_fused_features = fused_features
        
        text_embed_dim = inputs_embeds.size(-1)
        if projected_fused_features.size(-1) != text_embed_dim:
            projection_layer = nn.Linear(projected_fused_features.size(-1), text_embed_dim).to(projected_fused_features.device)
            projected_fused_features = projection_layer(projected_fused_features)
        
        if len(image_token_positions) > 0:
            first_image_pos = image_token_positions[0]
            if projected_fused_features.size(1) <= inputs_embeds.size(1) - first_image_pos:
                inputs_embeds[:, first_image_pos:first_image_pos+projected_fused_features.size(1)] = projected_fused_features
    
    outputs = model(inputs_embeds=inputs_embeds, labels=labels)
    
    return outputs.loss

def generate_prediction(model, tokenizer, input_ids, fused_features, image):
    """生成预测答案"""
    try:
        # 准备输入
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        
        # 替换图像token的嵌入为融合特征
        from models.llava.llava.constants import IMAGE_TOKEN_INDEX
        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        
        if image_token_mask.any():
            image_token_positions = torch.nonzero(image_token_mask, as_tuple=True)[1]
            
            if hasattr(model, 'mm_projector'):
                projected_fused_features = model.mm_projector(fused_features)
            else:
                projected_fused_features = fused_features
            
            text_embed_dim = inputs_embeds.size(-1)
            if projected_fused_features.size(-1) != text_embed_dim:
                projection_layer = nn.Linear(projected_fused_features.size(-1), text_embed_dim).to(projected_fused_features.device)
                projected_fused_features = projection_layer(projected_fused_features)
            
            if len(image_token_positions) > 0:
                first_image_pos = image_token_positions[0]
                if projected_fused_features.size(1) <= inputs_embeds.size(1) - first_image_pos:
                    inputs_embeds[:, first_image_pos:first_image_pos+projected_fused_features.size(1)] = projected_fused_features
        
        # 生成预测
        with torch.no_grad():
            output_ids = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
            )
            
            # 解码预测结果
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            return prediction
    except Exception as e:
        print(f"生成预测时出错: {e}")
        return "预测失败"

if __name__ == "__main__":
    analyze_predictions()