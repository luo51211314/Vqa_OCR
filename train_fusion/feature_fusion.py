import torch
import torch.nn as nn
import numpy as np
import json
from feature_extraction import FeatureExtractor

class FeatureFusion(nn.Module):
    def __init__(self, config, llava_model=None, tokenizer=None, ocr_model=None):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config, llava_model=llava_model, tokenizer=tokenizer, ocr_model=ocr_model)
        self.projection_dim = config['model_config'].get('projection_dim', 4096)  # 投影维度
        
        # 初始化时，我们将从LLaVA模型中获取实际的隐藏层大小
        self.hidden_size = None
        self.initialized = False
        
        # 定义OCR文本嵌入的投影层
        # 这个将在initialize方法中根据实际的嵌入维度进行初始化
        self.ocr_text_projector = None
        
        # 初始化调试步骤计数器
        self.debug_step = 0
        
        # 对比损失权重参数，确保从配置中读取
        self.contrastive_loss_weight = config['model_config'].get('contrastive_loss_weight', 0.1)
        
        # 如果提供了模型和分词器，直接初始化
        if llava_model is not None and tokenizer is not None:
            self.initialize(llava_model=llava_model, tokenizer=tokenizer)
        
    def initialize(self, llava_model=None, tokenizer=None):
        """初始化特征提取器和融合模块
        
        Args:
            llava_model: 已初始化的LLaVA模型实例
            tokenizer: 已初始化的tokenizer实例
        """
        if not self.initialized:
            # 使用传入的模型和tokenizer，不再尝试加载新模型
            if llava_model is None or tokenizer is None:
                raise ValueError("必须从外部传入llava_model和tokenizer")
            
            # 初始化feature_extractor，传入llava_model和tokenizer
            self.feature_extractor.initialize(llava_model=llava_model, tokenizer=tokenizer)
            
            # 获取LLaVA模型的隐藏层大小
            self.hidden_size = llava_model.config.hidden_size
            
            # 初始化OCR文本嵌入的投影层
            self.ocr_text_projector = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU()
            )
            # 手动初始化权重和偏置
            nn.init.xavier_uniform_(self.ocr_text_projector[0].weight)
            if self.ocr_text_projector[0].bias is not None:
                nn.init.zeros_(self.ocr_text_projector[0].bias)
            
            # 移动到模型设备
            device = next(llava_model.parameters()).device
            self.to(device)
            
            self.initialized = True
        
        return self
    
    def forward(self, image):
        """前向传播，融合LLaVA特征和OCR特征"""
        
        self.debug_step += 1
        
        # 首先提取OCR特征（处理原始图像）
        ocr_results = self.feature_extractor.extract_ocr_features(image)
        
        # 然后提取LLaVA的视觉特征（处理原始图像）
        llava_features = self.feature_extractor.extract_llava_features(image)
        
        # 添加数值稳定性保障 - 防止nan值
        if torch.isnan(llava_features).any():
            print(f"步骤 {self.debug_step} - 警告: LLaVA特征中存在nan值")
            llava_features = torch.nan_to_num(llava_features, nan=1e-8)
        llava_features = torch.clamp(llava_features, min=-1e5, max=1e5)  # 限制数值范围
        
        # 检查是否为批次结果
        is_batch_results = isinstance(ocr_results, list)
        batch_size = len(ocr_results) if is_batch_results else 1
        
        # 准备存储所有OCR嵌入的列表
        ocr_embeddings = []
        
        if is_batch_results:
            # 处理批次OCR结果
            for i, ocr_result in enumerate(ocr_results):
                ocr_text = ocr_result.get('full_description', '')
                
                # 将OCR文本描述转换为嵌入向量
                text_embedding = self.feature_extractor.text_to_embedding(ocr_text)
                ocr_embeddings.append(text_embedding)
            
            # 堆叠所有嵌入
            text_embedding = torch.cat(ocr_embeddings, dim=0)
        else:
            # 处理单个OCR结果
            ocr_result = ocr_results
            
            # 将OCR文本描述转换为嵌入向量
            text_embedding = self.feature_extractor.text_to_embedding(ocr_result['full_description'])
        
        # 投影层处理
        # 如果没有OCR文本，创建零向量
        if text_embedding.shape[0] == 0 or (not is_batch_results and not ocr_result['ocr_texts']):
            # 创建与llava_features兼容形状的零向量
            batch_size = llava_features.shape[0]
            ocr_features = torch.zeros((batch_size, self.hidden_size), 
                                       device=llava_features.device, 
                                       dtype=llava_features.dtype)
            print(f"步骤 {self.debug_step} - OCR does not find text, use zero vector")
        else:
            # 添加数值稳定性保障
            if torch.isnan(text_embedding).any():
                print(f"步骤 {self.debug_step} - 警告: 文本嵌入中存在nan值")
                text_embedding = torch.nan_to_num(text_embedding, nan=1e-8)
            text_embedding = torch.clamp(text_embedding, min=-1e5, max=1e5)
            
            # 确保投影层使用与文本嵌入相同的数据类型
            if self.ocr_text_projector[0].weight.dtype != text_embedding.dtype:
                self.ocr_text_projector = self.ocr_text_projector.to(dtype=text_embedding.dtype)
            
            # 使用投影层将文本嵌入映射到与LLaVA特征相同的空间
            ocr_features = self.ocr_text_projector(text_embedding)
            
            # 添加数值稳定性保障
            if torch.isnan(ocr_features).any():
                print(f"步骤 {self.debug_step} - 警告: OCR特征投影后存在nan值")
                ocr_features = torch.nan_to_num(ocr_features, nan=1e-8)
            ocr_features = torch.clamp(ocr_features, min=-1e5, max=1e5)
        
        # 确保批次大小匹配
        if llava_features.shape[0] != ocr_features.shape[0]:
            print(f"步骤 {self.debug_step} - 警告: 批次大小不匹配 - LLaVA: {llava_features.shape[0]}, OCR: {ocr_features.shape[0]}")
            # 取较小的批次大小
            min_batch = min(llava_features.shape[0], ocr_features.shape[0])
            llava_features = llava_features[:min_batch]
            ocr_features = ocr_features[:min_batch]
        
        # 特征融合 - 拼接
        # 为了匹配维度，我们需要调整ocr_features的形状
        try:
            if len(llava_features.shape) == 3:
                # 对于3D特征（batch_size, num_patches, hidden_size）
                ocr_features_expanded = ocr_features.unsqueeze(1)
                fused_features = torch.cat([ocr_features_expanded, llava_features], dim=1)
            elif len(llava_features.shape) == 2:
                # 对于2D特征（batch_size, hidden_size）
                llava_features_expanded = llava_features.unsqueeze(1)
                ocr_features_expanded = ocr_features.unsqueeze(1)
                fused_features = torch.cat([llava_features_expanded, ocr_features_expanded], dim=1)
            else:
                raise ValueError(f"不支持的特征形状: {llava_features.shape}")
                
        except Exception as e:
            print(f"步骤 {self.debug_step} - 特征融合失败: {e}")
            # 退化为使用LLaVA特征
            fused_features = llava_features
        
        # 添加数值稳定性保障
        if torch.isnan(fused_features).any():
            print(f"步骤 {self.debug_step} - 警告: 融合特征中存在nan值")
            fused_features = torch.nan_to_num(fused_features, nan=1e-8)
        fused_features = torch.clamp(fused_features, min=-1e5, max=1e5)
        
        # 计算轻量化对比损失
        contrastive_loss = self._compute_contrastive_loss(llava_features, ocr_features)
        
        return {
            "fused_features": fused_features,
            "llava_features": llava_features,
            "ocr_features": ocr_features,
            "ocr_result": ocr_result,
            "contrastive_loss": contrastive_loss
        }
    
    def _compute_contrastive_loss(self, llava_features, ocr_features):
        """计算轻量化的LLaVA和OCR特征对比损失
        
        使用余弦相似度作为对比损失，鼓励LLaVA和OCR特征在语义空间中接近
        """
        if llava_features.shape[0] == 0 or ocr_features.shape[0] == 0:
            return torch.tensor(0.0, device=llava_features.device) if llava_features.numel() > 0 else torch.tensor(0.0)
        
        # 确保批次大小匹配
        batch_size = min(llava_features.shape[0], ocr_features.shape[0])
        llava_features = llava_features[:batch_size]
        ocr_features = ocr_features[:batch_size]
        
        # 归一化特征向量
        llava_features_norm = nn.functional.normalize(llava_features, dim=-1)
        ocr_features_norm = nn.functional.normalize(ocr_features, dim=-1)
        
        # 计算余弦相似度
        cosine_sim = torch.sum(llava_features_norm * ocr_features_norm, dim=-1)
        
        # 对比损失：鼓励相似度接近1
        # 使用MSE损失，目标是相似度等于1
        loss = nn.functional.mse_loss(cosine_sim, torch.ones_like(cosine_sim))
        
        # 直接返回损失值，权重应用在模型中统一处理
        return loss

class FusedVisionProjector(nn.Module):
    def __init__(self, config, llava_model=None, tokenizer=None, ocr_model=None):
        super().__init__()
        self.config = config
        self.feature_fusion = FeatureFusion(config, llava_model=llava_model, tokenizer=tokenizer, ocr_model=ocr_model)
        
        # 初始化时，我们将从LLaVA模型中获取实际的投影层
        self.fusion_lm_projector = None
        self.fusion_projector = None  # 新增：融合特征投影层
        self.hidden_size = None
        self.initialized = False
        
        # 初始化调试步骤计数器
        self.debug_step = 0
        
        # 如果提供了模型和分词器，直接初始化
        if llava_model is not None and tokenizer is not None:
            self.initialize(llava_model=llava_model, tokenizer=tokenizer)
        
    def initialize(self, llava_model=None, tokenizer=None):
        """使用LLaVA模型的投影层初始化"""
        if not self.initialized:
            # 检查必要的模型是否已传入
            if llava_model is None or tokenizer is None:
                raise ValueError("必须从外部传入llava_model和tokenizer")
                
            # 初始化特征融合模块
            self.feature_fusion.initialize(llava_model=llava_model, tokenizer=tokenizer)
            
            # 获取LLaVA模型的投影层和隐藏层大小
            self.hidden_size = llava_model.config.hidden_size
            
            # 确保特征融合模块和投影层在同一设备上
            device = next(llava_model.parameters()).device
            self.to(device)
            
            # 初始化融合特征投影层
            # 融合特征在forward函数中会被展平为8192维度
            fusion_lm_projector_dim = self.hidden_size  # 输出维度等于hidden_size
            fusion_input_dim = 2 * self.hidden_size  # 展平后的特征维度
            
            # 创建融合特征投影层，输入为fusion_feature的维度，输出为hidden_size
            self.fusion_projector = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_lm_projector_dim, device=device),
                nn.GELU()
            )
            # 初始化权重
            nn.init.xavier_uniform_(self.fusion_projector[0].weight)
            if self.fusion_projector[0].bias is not None:
                nn.init.zeros_(self.fusion_projector[0].bias)
            
            # 创建fusion_lm_projector，输入为fusion投影后的维度，输出为hidden_size
            self.fusion_lm_projector = nn.Sequential(
                nn.Linear(fusion_lm_projector_dim, self.hidden_size, device=device),
                nn.GELU()
            )
            # 初始化权重
            nn.init.xavier_uniform_(self.fusion_lm_projector[0].weight)
            if self.fusion_lm_projector[0].bias is not None:
                nn.init.zeros_(self.fusion_lm_projector[0].bias)
            
            self.initialized = True
        
        return self
        
    def forward(self, image):
        """前向传播，融合特征并通过投影层"""
        if not self.initialized:
            raise RuntimeError("FusedVisionProjector尚未初始化，请确保在实例化时提供llava_model和tokenizer参数")
        
        self.debug_step += 1
        
        # 获取融合特征
        fusion_result = self.feature_fusion.forward(image)
        fused_features = fusion_result['fused_features']
        
        # 再次检查融合特征中的nan值
        if torch.isnan(fused_features).any():
            print(f"投影器步骤 {self.debug_step} - 警告: 进入投影层前的融合特征中存在nan值")
            fused_features = torch.nan_to_num(fused_features, nan=1e-8)
        
        # 通过投影层
        # 使用持久化的fusion_projector进行维度调整
        # 确保fusion_projector与特征在同一设备
        if self.fusion_projector[0].weight.device != fused_features.device:
            self.fusion_projector = self.fusion_projector.to(fused_features.device)
        
        # 展平特征：将bs*24096展平为bs*8192
        # 先获取原始形状
        batch_size = fused_features.size(0)
        # 展平为bs*8192
        fused_features_flattened = fused_features.view(batch_size, 8192)
        
        # 应用融合特征投影层
        fused_features_adapted = self.fusion_projector(fused_features_flattened)
        
        # 检查投影后的nan值
        if torch.isnan(fused_features_adapted).any():
            print(f"投影器步骤 {self.debug_step} - 警告: 融合投影层输出中存在nan值")
            fused_features_adapted = torch.nan_to_num(fused_features_adapted, nan=1e-8)
        
        # 使用自创的fusion_lm_projector
        try:
            # 确保fusion_lm_projector与特征在同一设备
            if self.fusion_lm_projector[0].weight.device != fused_features_adapted.device:
                self.fusion_lm_projector = self.fusion_lm_projector.to(fused_features_adapted.device)
            
            # 使用自创的fusion_lm_projector
            projected_features = self.fusion_lm_projector(fused_features_adapted)
            
            # 检查最终投影后的nan值
            if torch.isnan(projected_features).any():
                print(f"投影器步骤 {self.debug_step} - 警告: 最终投影特征中存在nan值")
                projected_features = torch.nan_to_num(projected_features, nan=1e-8)
        except Exception as e:
            # 尝试使用更稳定的投影方法
            with torch.autocast(device_type='cuda', enabled=False):
                projected_features = self.fusion_lm_projector(fused_features_adapted.float())
                if torch.isnan(projected_features).any():
                    print(f"投影器步骤 {self.debug_step} - 警告: 替代投影方法后仍存在nan值")
                    projected_features = torch.nan_to_num(projected_features, nan=1e-8)
        
        # 最终检查nan值
        if torch.isnan(projected_features).any():
            print(f"投影器步骤 {self.debug_step} - 严重警告: 最终特征中仍存在nan值")
            projected_features = torch.nan_to_num(projected_features, nan=1e-8)
        
        # 更新结果字典
        fusion_result['projected_features'] = projected_features
        
        return fusion_result

if __name__ == "__main__":
    # 测试特征融合模块
    with open("config.json", "r") as f:
        config = json.load(f)
    
    fusion_module = FeatureFusion(config)
    fusion_module.initialize()
    
    print("特征融合模块初始化成功")