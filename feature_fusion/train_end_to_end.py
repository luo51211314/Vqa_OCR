import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import sys
import pandas as pd
from PIL import Image
import json
import io
from tqdm import tqdm
# 添加混合精度训练支持
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_fusion.contrastive_alignment import ContrastiveAlignment
from feature_fusion.attention_fusion import AttentionFusion
from feature_fusion.real_feature_extractor import RealFeatureExtractor
from model_loader.loader_llava import LLaVALoader

class VQADataset(torch.utils.data.Dataset):
    """VQA数据集类"""
    
    def __init__(self, data_path, image_folder, tokenizer, image_processor, max_length=128, device="cuda"):  # 默认max_length改为128
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.device = device
        
        # 加载数据
        if data_path.endswith('.parquet'):
            self.data = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError("不支持的数据格式，请使用parquet或json格式")
        
        # 特征提取器
        self.feature_extractor = RealFeatureExtractor(device=device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx] if hasattr(self.data, 'iloc') else self.data[idx]
        
        # 获取图像数据
        if 'image_bytes' in item:
            # 从字节数据加载图像
            image_bytes = item['image_bytes']
            if isinstance(image_bytes, bytes):
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                # 如果image_bytes不是bytes类型，尝试从文件路径加载
                image_path = os.path.join(self.image_folder, item.get('image', ''))
                image = Image.open(image_path).convert('RGB')
        else:
            # 从文件路径加载图像
            image_path = os.path.join(self.image_folder, item['image'])
            image = Image.open(image_path).convert('RGB')
        
        # 获取问题和答案
        question = item['question']
        answer = item['answer']
        
        # 处理答案格式：如果是numpy数组或列表，提取第一个元素
        if hasattr(answer, '__iter__') and not isinstance(answer, str):
            # 如果是可迭代对象但不是字符串，尝试提取第一个元素
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        
        # 确保答案是字符串类型
        answer = str(answer) if answer is not None else ""
        
        # 处理图像和文本
        processed_image = self.image_processor(image, return_tensors='pt')['pixel_values'].squeeze(0)
        
        # 构建输入文本 - 简化指令，要求直接输出简短答案
        input_text = f"<image>\n{question}\nPlease answer directly without explanation:"
        
        # 构建标签文本 - 使用更简短的答案格式
        label_text = answer
        
        # 关键修复：使用LLaVA的tokenizer_image_token函数正确处理图像token
        from models.llava.llava.constants import IMAGE_TOKEN_INDEX
        from models.llava.llava.mm_utils import tokenizer_image_token
        
        # 使用tokenizer_image_token处理输入文本
        input_ids = tokenizer_image_token(
            input_text, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).squeeze(0)
        
        # 对标签文本使用普通tokenizer，限制最大长度
        label_ids = self.tokenizer.encode(
            label_text, 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length'
        )
        
        # 确保input_ids长度不超过max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            # 如果长度不足，进行padding
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
        
        # 转换为tensor
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        
        # 提取特征 - 限制特征序列长度以节省内存
        with torch.no_grad():
            # 提取LLaVA特征，限制序列长度
            llava_features = self.feature_extractor.extract_llava_features(
                processed_image.unsqueeze(0), None
            )
            
            # 提取OCR特征，限制序列长度
            ocr_features = self.feature_extractor.extract_features(
                processed_image.unsqueeze(0), 'paddleocr'
            )
            
            # 如果特征序列过长，进行降采样
            max_feature_length = 64  # 限制特征序列最大长度
            if llava_features.size(1) > max_feature_length:
                llava_features = llava_features[:, :max_feature_length, :]
            if ocr_features.size(1) > max_feature_length:
                ocr_features = ocr_features[:, :max_feature_length, :]
        
        return {
            'image': processed_image,
            'input_ids': input_ids,
            'labels': label_ids,
            'llava_features': llava_features.squeeze(0),
            'ocr_features': ocr_features.squeeze(0),
            'question': question,
            'answer': answer
        }

def compute_vqa_loss(model, input_ids, labels, fused_features, images):
    """计算VQA损失，使用融合特征替换图像token的嵌入"""
    
    # 关键修复：确保input_ids中的索引都在有效范围内
    # 首先检查input_ids的最小值和最大值
    min_idx = input_ids.min().item()
    max_idx = input_ids.max().item()
    # print(f"DEBUG: input_ids索引范围: [{min_idx}, {max_idx}]")
    
    # 关键修复：处理负索引问题
    # 如果input_ids包含负索引，需要将其映射到有效范围内
    if min_idx < 0:
        # print(f"DEBUG: 检测到负索引，需要映射到有效范围")
        
        # 获取词汇表大小
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000
        # print(f"DEBUG: 词汇表大小: {vocab_size}")
        
        # 创建一个映射：将负索引映射到词汇表末尾的保留位置
        # 假设词汇表末尾有一些保留位置可用于特殊token
        reserved_start = vocab_size - 100  # 保留最后100个位置给特殊token
        
        # 创建input_ids的副本，将负索引映射到保留位置
        valid_input_ids = input_ids.clone()
        
        # 找到所有负索引
        negative_mask = input_ids < 0
        if negative_mask.any():
            # 将负索引映射到保留位置
            # 例如：-200 -> reserved_start + 0, -199 -> reserved_start + 1, 等等
            negative_indices = input_ids[negative_mask]
            unique_negatives = torch.unique(negative_indices)
            
            # 为每个唯一的负索引分配一个保留位置
            mapping_dict = {}
            for i, neg_idx in enumerate(unique_negatives):
                mapping_dict[neg_idx.item()] = reserved_start + i
            
            # 应用映射
            for neg_idx, mapped_idx in mapping_dict.items():
                valid_input_ids[input_ids == neg_idx] = mapped_idx
            
            # print(f"DEBUG: 负索引映射完成: {mapping_dict}")
        else:
            print("DEBUG: 没有检测到负索引，使用原始input_ids")
            valid_input_ids = input_ids
    else:
        valid_input_ids = input_ids
    
    # 准备输入（使用处理后的有效索引）
    inputs_embeds = model.model.get_input_embeddings()(valid_input_ids)
    
    # 替换图像token的嵌入为融合特征
    from models.llava.llava.constants import IMAGE_TOKEN_INDEX
    
    # 关键修复：改进图像token检测逻辑
    # 使用处理后的有效索引进行检测
    if IMAGE_TOKEN_INDEX < 0:
        # 检查IMAGE_TOKEN_INDEX是否被映射到了有效位置
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000
        reserved_start = vocab_size - 100
        
        # 计算IMAGE_TOKEN_INDEX对应的有效位置
        # 假设负索引按顺序映射到保留位置
        image_token_valid_idx = reserved_start + abs(IMAGE_TOKEN_INDEX) % 100
        
        # 使用映射后的位置检测图像token
        image_token_mask = (valid_input_ids == image_token_valid_idx)
        
        if image_token_mask.any():
            image_token_positions = torch.nonzero(image_token_mask, as_tuple=True)[1]
        else:
            # 如果没有检测到图像token，使用默认位置
            image_token_positions = torch.tensor([0], device=input_ids.device)
    else:
        # 如果IMAGE_TOKEN_INDEX是正常值，使用原来的检测方法
        image_token_mask = (valid_input_ids == IMAGE_TOKEN_INDEX)
        
        if image_token_mask.any():
            image_token_positions = torch.nonzero(image_token_mask, as_tuple=True)[1]
        else:
            # 如果没有检测到图像token，使用默认位置
            image_token_positions = torch.tensor([0], device=input_ids.device)
    
    # 关键修复：优化内存使用 - 减少融合特征的序列长度
    # 如果融合特征序列过长，进行适当的降采样
    max_fusion_length = 128  # 限制融合特征的最大长度
    if fused_features.size(1) > max_fusion_length:
        # 使用平均池化进行降采样
        pool_factor = fused_features.size(1) // max_fusion_length
        if pool_factor > 1:
            fused_features = torch.nn.functional.avg_pool1d(
                fused_features.transpose(1, 2), 
                kernel_size=pool_factor, 
                stride=pool_factor
            ).transpose(1, 2)
    
    if len(image_token_positions) > 0:
        # ===== 关键修复：确保LLaVA投影层始终参与梯度传播 =====
        
        # 修复1：正确访问LLaVA的mm_projector，确保梯度流
        # LLaVA模型通过get_model()方法访问内部模型
        mm_projector_found = False
        projected_fused_features = None
        
        # 尝试多种访问路径来找到mm_projector
        if hasattr(model, 'get_model'):
            internal_model = model.get_model()
            if hasattr(internal_model, 'mm_projector'):
                projected_fused_features = internal_model.mm_projector(fused_features)
                mm_projector_found = True
                print(f"使用LLaVA mm_projector (通过get_model): {fused_features.shape} -> {projected_fused_features.shape}")
        
        if not mm_projector_found and hasattr(model, 'model'):
            if hasattr(model.model, 'mm_projector'):
                projected_fused_features = model.model.mm_projector(fused_features)
                mm_projector_found = True
                print(f"使用LLaVA mm_projector (通过model.model): {fused_features.shape} -> {projected_fused_features.shape}")
        
        if not mm_projector_found and hasattr(model, 'mm_projector'):
            projected_fused_features = model.mm_projector(fused_features)
            mm_projector_found = True
            print(f"使用LLaVA mm_projector (直接访问): {fused_features.shape} -> {projected_fused_features.shape}")
        
        if not mm_projector_found:
            # 如果没有mm_projector，直接使用融合特征（但这种情况不应该发生）
            projected_fused_features = fused_features
            print("警告：未找到mm_projector，直接使用融合特征")
        
        # 修复2：确保投影后的特征维度与文本嵌入维度匹配
        text_embed_dim = inputs_embeds.size(-1)
        
        if projected_fused_features.size(-1) != text_embed_dim:
            print(f"维度不匹配：投影特征 {projected_fused_features.size(-1)} vs 文本嵌入 {text_embed_dim}")
            # 如果维度不匹配，使用可学习的线性投影（带梯度）
            projection_layer = nn.Linear(projected_fused_features.size(-1), text_embed_dim).to(projected_fused_features.device)
            projected_fused_features = projection_layer(projected_fused_features)
            print(f"使用额外投影层: {projected_fused_features.size(-1)} -> {text_embed_dim}")
        
        # 修复3：使用in-place操作保持梯度流
        first_image_pos = image_token_positions[0]
        
        # 计算可替换的长度
        replaceable_length = inputs_embeds.size(1) - first_image_pos
        required_length = projected_fused_features.size(1)
        
        # 确保替换的特征维度正确，保持梯度流
        if required_length <= replaceable_length:
            # 使用in-place操作，保持梯度连接
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, first_image_pos:first_image_pos+required_length] = projected_fused_features
            print(f"特征替换成功：位置 {first_image_pos}-{first_image_pos+required_length}")
        else:
            # 如果长度不匹配，截断融合特征但保持梯度
            truncated_features = projected_fused_features[:, :replaceable_length]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, first_image_pos:] = truncated_features
            print(f"特征截断替换：位置 {first_image_pos}-{inputs_embeds.size(1)}")

    # 关键修复：优化内存使用 - 使用梯度检查点或减少计算图复杂度
    # 前向传播 - 修复：手动实现损失计算
    try:
        # 修复：只使用LLaVA的语言模型部分，避免重复的视觉处理
        # 获取隐藏状态
        hidden_states = model.model(inputs_embeds=inputs_embeds).last_hidden_state
        
        # 使用lm_head计算logits
        logits = model.lm_head(hidden_states)
        
        # 手动计算交叉熵损失
        # 将logits和labels展平
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算交叉熵损失
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
            ignore_index=-100  # 忽略padding token
        )
        
        return loss
    except torch.cuda.OutOfMemoryError as e:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 尝试使用更小的批次或特征
        # 进一步减少融合特征的长度
        reduced_fused_features = fused_features[:, :64]  # 只使用前64个token
        
        # 重新计算投影 - 使用相同的访问逻辑
        mm_projector_found = False
        reduced_projected = None
        
        if hasattr(model, 'get_model'):
            internal_model = model.get_model()
            if hasattr(internal_model, 'mm_projector'):
                reduced_projected = internal_model.mm_projector(reduced_fused_features)
                mm_projector_found = True
        
        if not mm_projector_found and hasattr(model, 'model'):
            if hasattr(model.model, 'mm_projector'):
                reduced_projected = model.model.mm_projector(reduced_fused_features)
                mm_projector_found = True
        
        if not mm_projector_found and hasattr(model, 'mm_projector'):
            reduced_projected = model.mm_projector(reduced_fused_features)
            mm_projector_found = True
        
        if not mm_projector_found:
            reduced_projected = reduced_fused_features
        
        if reduced_projected.size(-1) != text_embed_dim:
            projection_layer = nn.Linear(reduced_projected.size(-1), text_embed_dim).to(reduced_projected.device)
            reduced_projected = projection_layer(reduced_projected)
        
        # 重新进行特征替换
        inputs_embeds_reduced = model.model.get_input_embeddings()(valid_input_ids)
        replaceable_length = inputs_embeds_reduced.size(1) - first_image_pos
        required_length = min(reduced_projected.size(1), replaceable_length)
        
        if required_length > 0:
            inputs_embeds_reduced = inputs_embeds_reduced.clone()
            inputs_embeds_reduced[:, first_image_pos:first_image_pos+required_length] = reduced_projected[:, :required_length]
        
        # 修复：再次尝试前向传播 - 手动实现损失计算
        hidden_states = model.model(inputs_embeds=inputs_embeds_reduced).last_hidden_state
        logits = model.lm_head(hidden_states)
        
        # 手动计算交叉熵损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss

def train_end_to_end():
    """端到端训练函数"""
    parser = argparse.ArgumentParser(description='端到端训练：对齐+融合+微调LLaVA投影矩阵')
    parser.add_argument('--expert_model', type=str, default='paddleocr', 
                       choices=['paddleocr', 'pix2struct'], 
                       help='专家模型类型')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/dataset/trainingVQA', 
                       help='数据目录')
    parser.add_argument('--weight_dir', type=str, default='/root/autodl-tmp/weight', 
                       help='权重保存目录')
    parser.add_argument('--alpha', type=float, default=0.8, help='对比损失权重')
    parser.add_argument('--beta', type=float, default=0.2, help='VQA损失权重')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')  # 从256减小到128
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='训练设备')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--use_amp', action='store_true', default=True, help='启用混合精度训练')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    
    args = parser.parse_args()
    
    # 验证权重参数
    if args.alpha + args.beta != 1.0:
        raise ValueError("alpha + beta 必须等于 1.0")
    
    # 创建权重目录
    os.makedirs(args.weight_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 设置PyTorch内存优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 初始化混合精度训练
    scaler = GradScaler(enabled=args.use_amp)
    
    # 加载LLaVA模型
    print("加载LLaVA模型...")
    llava_loader = LLaVALoader()
    # 需要先调用load_model方法加载模型
    model_path = "/root/autodl-tmp/model/llava_hug"  # 修正后的模型路径
    llava_tokenizer, llava_model, llava_image_processor, context_len = llava_loader.load_model(model_path, device=args.device)
    
    # 冻结LLaVA的语言模型参数，只训练投影矩阵
    for name, param in llava_model.named_parameters():
        # 关键修复：强制可训练参数为 FP32，否则 AMP 无法反缩放
        if 'mm_projector' in name:
            param.data = param.data.to(torch.float32)   # 新增
            param.requires_grad = True
            print(f"设置参数为可训练: {name}, 形状: {param.shape}")
        else:
            param.requires_grad = False
    
    # 验证哪些参数是可训练的
    trainable_param_names = [name for name, param in llava_model.named_parameters() if param.requires_grad]
    print(f"可训练参数数量: {len(trainable_param_names)}")
    for name in trainable_param_names:
        print(f"  - {name}")
    
    # 数据集
    print("加载数据集...")
    train_dataset = VQADataset(
        data_path=os.path.join(args.data_dir, 'train_contrastive_fusion_train.parquet'),
        image_folder=args.data_dir,
        tokenizer=llava_tokenizer,
        image_processor=llava_image_processor,
        max_length=args.max_length,  # 使用降低后的序列长度
        device=args.device
    )
    
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 对齐模块 - 修复维度配置
    alignment_module = ContrastiveAlignment(
        llava_hidden_size=1024,  # 修正：实际LLaVA特征维度
        expert_hidden_size=1024,  # 修正：实际专家特征维度
        projection_dim=1024,  # 对齐后的维度，与融合模块输入匹配
        temperature=0.07  # 进一步降低温度参数以提高稳定性
    ).to(device)
    
    # 融合模块 - 修复维度配置
    fusion_module = AttentionFusion(
        llava_hidden_size=1024,  # 修正：与对齐模块输出匹配
        expert_hidden_size=1024,  # 修正：与对齐模块输出匹配
        projection_dim=1024,  # 融合输出维度，与LLaVA投影层输入匹配
        num_heads=8  # 增加注意力头数以提高表达能力
    ).to(device)
    
    # 明确获取可训练参数
    alignment_params = list(alignment_module.parameters())
    fusion_params = list(fusion_module.parameters())
    
    # 修复LLaVA投影层梯度问题 - 确保投影层可训练
    llava_proj_params = []
    for name, param in llava_model.named_parameters():
        if 'mm_projector' in name:
            param.requires_grad = True
            llava_proj_params.append(param)
            print(f"LLaVA投影层参数 {name} 设置为可训练")
    
    print(f"对齐模块参数数量: {len(alignment_params)}")
    print(f"融合模块参数数量: {len(fusion_params)}")
    print(f"LLaVA投影参数数量: {len(llava_proj_params)}")
    
    # 合并所有可训练参数
    trainable_params = alignment_params + fusion_params + llava_proj_params
    
    # 极大降低学习率以提高训练稳定性
    optimizer = optim.AdamW(trainable_params, lr=1e-7, weight_decay=0.01, eps=1e-8)  # 极大降低学习率
    
    # 学习率调度器 - 使用更温和的策略
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader), eta_min=1e-10)
    
    # 梯度裁剪阈值极大降低以防止梯度爆炸
    gradient_clip_value = 0.001  # 极大降低梯度裁剪阈值
    
    # 训练循环
    alignment_module.train()
    fusion_module.train()
    llava_model.train()
    
    # 损失记录
    loss_history = {
        'epoch': [],
        'total_loss': [],
        'contrastive_loss': [],
        'vqa_loss': [],
        'learning_rates': []
    }
    
    # 梯度范数记录
    grad_norm_history = {
        'alignment': [],
        'fusion': [],
        'llava_proj': []
    }
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_contrastive_loss = 0
        total_vqa_loss = 0
        total_grad_norm_alignment = 0
        total_grad_norm_fusion = 0
        total_grad_norm_llava_proj = 0
        
        # 创建进度条
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据并移动到设备
            images = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            llava_features = batch['llava_features'].to(device, non_blocking=True)
            ocr_features = batch['ocr_features'].to(device, non_blocking=True)
            
            # 调试信息：打印特征形状
            if args.debug and batch_idx == 0 and epoch == 0:
                print(f"LLaVA特征形状: {llava_features.shape}")
                print(f"OCR特征形状: {ocr_features.shape}")
                print(f"输入ID形状: {input_ids.shape}")
                print(f"标签形状: {labels.shape}")
            
            # 使用混合精度训练
            with autocast(enabled=args.use_amp):
                # 1. 对比学习对齐
                aligned_llava, aligned_expert, contrastive_loss = alignment_module(
                    llava_features, ocr_features, labels=labels, mode="train"
                )
                
                # 调试信息：检查对齐后的特征
                if args.debug and batch_idx == 0 and epoch == 0:
                    print(f"对齐后LLaVA特征形状: {aligned_llava.shape}")
                    print(f"对齐后OCR特征形状: {aligned_expert.shape}")
                    if contrastive_loss is not None:
                        print(f"对比损失值: {contrastive_loss.item()}")
                
                # 2. 注意力融合
                fused_features = fusion_module(aligned_llava, aligned_expert)
                
                # 调试信息：检查融合后的特征
                if args.debug and batch_idx == 0 and epoch == 0:
                    print(f"融合后特征形状: {fused_features.shape}")
                
                # 3. 计算VQA损失
                vqa_loss = compute_vqa_loss(llava_model, input_ids, labels, fused_features, images)
                
                # 调试：检查融合特征是否包含梯度信息
                if args.debug and batch_idx % 10 == 0:
                    print(f"融合特征是否需要梯度: {fused_features.requires_grad}")
                    if fused_features.grad_fn is not None:
                        print(f"融合特征梯度函数: {fused_features.grad_fn}")
                    else:
                        print("融合特征没有梯度函数")
                
                # 4. 关键修复：确保对比损失也能传播到LLaVA投影层
                # 通过融合特征间接传播对比损失的梯度
                if contrastive_loss is not None and contrastive_loss.item() > 0:
                    # 创建一个辅助损失，通过融合特征传播对比损失的梯度
                    # 使用融合特征的均值和对比损失构建一个额外的梯度传播路径
                    fusion_mean = fused_features.mean()
                    contrastive_gradient_prop = contrastive_loss * fusion_mean * 0.001  # 小权重避免干扰主损失
                    
                    # 将对比损失的梯度通过融合特征传播到LLaVA投影层
                    total_loss = args.alpha * contrastive_loss + args.beta * vqa_loss + contrastive_gradient_prop
                else:
                    total_loss = args.alpha * contrastive_loss + args.beta * vqa_loss
                
                loss = total_loss
            
            # 反向传播
            optimizer.zero_grad()
            
            # ===== 关键诊断测试：分离损失反向传播 =====
            if args.debug and batch_idx % 10 == 0:
                print(f"\n=== 诊断测试：分离损失反向传播 ===")
                
                # 测试1：只反向传播对比损失
                if contrastive_loss is not None and contrastive_loss.item() > 0:
                    print("测试1：只反向传播对比损失...")
                    optimizer.zero_grad()
                    scaler.scale(contrastive_loss).backward(retain_graph=True)
                    
                    # 检查对比损失是否能传播到LLaVA投影层
                    llava_has_grad_contrastive = False
                    for name, param in llava_model.named_parameters():
                        if 'mm_projector' in name and param.requires_grad:
                            if param.grad is not None and param.grad.norm().item() > 0:
                                llava_has_grad_contrastive = True
                                print(f"  对比损失 -> LLaVA投影 {name}: 梯度范数 = {param.grad.norm().item():.6f}")
                    
                    if not llava_has_grad_contrastive:
                        print("  对比损失无法传播到LLaVA投影层！")
                    
                    optimizer.zero_grad()  # 清空梯度
                
                # 测试2：只反向传播VQA损失
                if vqa_loss is not None and vqa_loss.item() > 0:
                    print("测试2：只反向传播VQA损失...")
                    optimizer.zero_grad()
                    scaler.scale(vqa_loss).backward(retain_graph=True)
                    
                    # 检查VQA损失是否能传播到LLaVA投影层
                    llava_has_grad_vqa = False
                    for name, param in llava_model.named_parameters():
                        if 'mm_projector' in name and param.requires_grad:
                            if param.grad is not None and param.grad.norm().item() > 0:
                                llava_has_grad_vqa = True
                                print(f"  VQA损失 -> LLaVA投影 {name}: 梯度范数 = {param.grad.norm().item():.6f}")
                    
                    if not llava_has_grad_vqa:
                        print("  VQA损失无法传播到LLaVA投影层！")
                    
                    optimizer.zero_grad()  # 清空梯度
                
                # 测试3：检查融合特征的梯度流
                print("测试3：检查融合特征梯度流...")
                if fused_features.requires_grad:
                    print(f"  融合特征 requires_grad: True")
                    if fused_features.grad_fn is not None:
                        print(f"  融合特征 grad_fn: {fused_features.grad_fn}")
                        # 尝试手动反向传播到融合特征
                        try:
                            optimizer.zero_grad()
                            test_loss = fused_features.sum()
                            scaler.scale(test_loss).backward(retain_graph=True)
                            print("  融合特征可以反向传播！")
                        except Exception as e:
                            print(f"  融合特征反向传播失败: {e}")
                    else:
                        print("  融合特征没有grad_fn！")
            
            # 使用混合精度反向传播（原始方式）
            scaler.scale(loss).backward()
            
            # 检查是否有梯度 - 保留梯度检查计算打印
            if args.debug and batch_idx % 10 == 0:
                print("\n=== 梯度检查计算 ===")
                
                # 检查融合模块的梯度
                fusion_has_grad = False
                for name, param in fusion_module.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 0:
                            fusion_has_grad = True
                            print(f"融合模块 {name}: 梯度范数 = {grad_norm:.6f}")
                
                if not fusion_has_grad:
                    print("融合模块: 所有参数梯度为0")
                
                # 检查LLaVA投影层的梯度 - 修复检查逻辑
                llava_has_grad = False
                for name, param in llava_model.named_parameters():
                    if 'mm_projector' in name and param.requires_grad:
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if grad_norm > 0:
                                llava_has_grad = True
                                print(f"LLaVA投影 {name}: 梯度范数 = {grad_norm:.6f}")
                        else:
                            print(f"LLaVA投影 {name}: 梯度为None")
                
                if not llava_has_grad:
                    print("LLaVA投影: 所有参数梯度为0或None")
                
                # 检查对齐模块的梯度
                alignment_has_grad = False
                for name, param in alignment_module.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 0:
                            alignment_has_grad = True
                            print(f"对齐模块 {name}: 梯度范数 = {grad_norm:.6f}")
            
            # 计算梯度范数用于调试
            grad_norm_alignment = torch.nn.utils.clip_grad_norm_(alignment_module.parameters(), float('inf'))
            grad_norm_fusion = torch.nn.utils.clip_grad_norm_(fusion_module.parameters(), float('inf'))
            grad_norm_llava_proj = torch.nn.utils.clip_grad_norm_(llava_proj_params, float('inf'))
            
            # 特殊处理：对层归一化参数进行额外的梯度缩放
            layer_norm_scaled_count = 0
            for module in [alignment_module, fusion_module]:
                for name, param in module.named_parameters():
                    if 'layer_norm' in name and param.grad is not None:
                        # 对层归一化的梯度进行缩放
                        param.grad = param.grad / 1000.0  # 更大的缩放因子
                        layer_norm_scaled_count += 1
            
            if layer_norm_scaled_count > 0:
                print(f"已缩放 {layer_norm_scaled_count} 个层归一化参数的梯度")
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=gradient_clip_value)
            
            # 使用混合精度优化器步进
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()  # 每个batch后更新学习率
            
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item() if contrastive_loss is not None else 0
            total_vqa_loss += vqa_loss.item()
            total_grad_norm_alignment += grad_norm_alignment.item()
            total_grad_norm_fusion += grad_norm_fusion.item()
            total_grad_norm_llava_proj += grad_norm_llava_proj.item()
            
            # 更新进度条描述
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Contrastive': f'{contrastive_loss.item() if contrastive_loss else 0:.4f}',
                'VQA': f'{vqa_loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 调试信息：每10个batch打印一次梯度范数信息
            if args.debug and batch_idx % 10 == 0:
                print(f"\nEpoch {epoch+1}, Batch {batch_idx}:")
                print(f"  梯度范数 - 对齐模块: {grad_norm_alignment.item():.4f}")
                print(f"  梯度范数 - 融合模块: {grad_norm_fusion.item():.4f}")
                print(f"  梯度范数 - LLaVA投影: {grad_norm_llava_proj.item():.4f}")
            
            # 定期清理GPU缓存
            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 关闭进度条
        pbar.close()
        
        avg_loss = total_loss / len(dataloader)
        avg_contrastive = total_contrastive_loss / len(dataloader)
        avg_vqa = total_vqa_loss / len(dataloader)
        avg_grad_norm_alignment = total_grad_norm_alignment / len(dataloader)
        avg_grad_norm_fusion = total_grad_norm_fusion / len(dataloader)
        avg_grad_norm_llava_proj = total_grad_norm_llava_proj / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录损失历史
        loss_history['epoch'].append(epoch + 1)
        loss_history['total_loss'].append(avg_loss)
        loss_history['contrastive_loss'].append(avg_contrastive)
        loss_history['vqa_loss'].append(avg_vqa)
        loss_history['learning_rates'].append(current_lr)
        
        # 记录梯度范数历史
        grad_norm_history['alignment'].append(avg_grad_norm_alignment)
        grad_norm_history['fusion'].append(avg_grad_norm_fusion)
        grad_norm_history['llava_proj'].append(avg_grad_norm_llava_proj)
        
        print(f'Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, '
              f'Avg Contrastive: {avg_contrastive:.4f}, Avg VQA: {avg_vqa:.4f}')
        print(f'Learning Rate: {current_lr:.2e}')
        print(f'Gradient Norms - Alignment: {avg_grad_norm_alignment:.4f}, '
              f'Fusion: {avg_grad_norm_fusion:.4f}, LLaVA Proj: {avg_grad_norm_llava_proj:.4f}')
        
        # 保存权重
        alignment_path = os.path.join(args.weight_dir, f'alignment_{args.expert_model}_epoch_{epoch+1}.pth')
        fusion_path = os.path.join(args.weight_dir, f'fusion_{args.expert_model}_epoch_{epoch+1}.pth')
        llava_path = os.path.join(args.weight_dir, f'llava_projection_{args.expert_model}_epoch_{epoch+1}.pth')
        
        torch.save(alignment_module.state_dict(), alignment_path)
        torch.save(fusion_module.state_dict(), fusion_path)
        
        # 只保存投影矩阵权重
        projection_state_dict = {}
        for name, param in llava_model.named_parameters():
            if 'mm_projector' in name:
                projection_state_dict[name] = param.data.cpu()
        torch.save(projection_state_dict, llava_path)
        
        print(f'Alignment weights saved to: {alignment_path}')
        print(f'Fusion weights saved to: {fusion_path}')
        print(f'LLaVA projection weights saved to: {llava_path}')
        
        # 绘制损失曲线
        if epoch > 0:  # 至少有两个点才能绘图
            plt.figure(figsize=(15, 5))
            
            # 总损失
            plt.subplot(1, 3, 1)
            plt.plot(loss_history['epoch'], loss_history['total_loss'], label='Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Total Loss')
            plt.grid(True)
            
            # 对比损失和VQA损失
            plt.subplot(1, 3, 2)
            plt.plot(loss_history['epoch'], loss_history['contrastive_loss'], label='Contrastive Loss')
            plt.plot(loss_history['epoch'], loss_history['vqa_loss'], label='VQA Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Individual Losses')
            plt.legend()
            plt.grid(True)
            
            # 学习率
            plt.subplot(1, 3, 3)
            plt.plot(loss_history['epoch'], loss_history['learning_rates'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
            
            plt.tight_layout()
            loss_plot_path = os.path.join(args.weight_dir, f'loss_curves_epoch_{epoch+1}.png')
            plt.savefig(loss_plot_path)
            plt.close()
            print(f'Loss curves saved to: {loss_plot_path}')
    
    # 保存损失历史
    import json
    loss_history_path = os.path.join(args.weight_dir, 'loss_history.json')
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f'Loss history saved to: {loss_history_path}')
    
    # 保存梯度范数历史
    grad_norm_history_path = os.path.join(args.weight_dir, 'grad_norm_history.json')
    with open(grad_norm_history_path, 'w') as f:
        json.dump(grad_norm_history, f, indent=2)
    print(f'Gradient norm history saved to: {grad_norm_history_path}')

if __name__ == "__main__":
    train_end_to_end()