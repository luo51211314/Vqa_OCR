import torch
import torch.nn as nn
import numpy as np
import json
from feature_extraction import FeatureExtractor

class FeatureFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.projection_dim = config['model_config'].get('projection_dim', 4096)  # 投影维度
        
        # 初始化时，我们将从LLaVA模型中获取实际的隐藏层大小
        self.hidden_size = None
        self.initialized = False
        
        # 定义OCR文本嵌入的投影层
        # 这个将在initialize方法中根据实际的嵌入维度进行初始化
        self.ocr_text_projector = None
        
        # 初始化调试步骤计数器
        self.debug_step = 0
        
    def initialize(self, llava_model=None, tokenizer=None):
        """初始化特征提取器和融合模块
        
        Args:
            llava_model: 已初始化的LLaVA模型实例，如果提供则传递给feature_extractor
            tokenizer: 已初始化的tokenizer实例，如果提供则传递给feature_extractor
        """
        if not self.initialized:
            # 将llava_model和tokenizer传递给feature_extractor以避免重复加载
            model, tokenizer, ocr_model = self.feature_extractor.initialize(
                llava_model=llava_model, 
                tokenizer=tokenizer
            )
            
            # 获取LLaVA模型的隐藏层大小
            self.hidden_size = model.config.hidden_size
            
            # 初始化OCR文本嵌入的投影层
            # 注意：text_to_embedding方法已经使用LLaVA的嵌入层，输出维度是hidden_size
            # 所以我们直接使用线性层作为特征转换，保持相同维度但可以学习不同的表示
            self.ocr_text_projector = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.xavier_uniform_(self.ocr_text_projector.weight)
            if self.ocr_text_projector.bias is not None:
                nn.init.zeros_(self.ocr_text_projector.bias)
            
            # 移动到模型设备
            device = next(model.parameters()).device
            self.to(device)
            
            self.initialized = True
        
        return self
    
    def forward(self, image):
        """前向传播，融合LLaVA特征和OCR特征"""
        if not self.initialized:
            self.initialize()
        
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
            if self.ocr_text_projector.weight.dtype != text_embedding.dtype:
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
        
        return {
            "fused_features": fused_features,
            "llava_features": llava_features,
            "ocr_features": ocr_features,
            "ocr_result": ocr_result
        }

class FusedVisionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_fusion = FeatureFusion(config)
        
        # 初始化时，我们将从LLaVA模型中获取实际的投影层
        self.vision_projector = None
        self.hidden_size = None
        self.initialized = False
        
        # 初始化调试步骤计数器
        self.debug_step = 0
        
    def initialize(self, llava_model=None, tokenizer=None):
        """使用LLaVA模型的投影层初始化"""
        if not self.initialized:
            # 初始化特征融合模块，传入llava_model和tokenizer以避免重复加载
            self.feature_fusion.initialize(
                llava_model=llava_model,
                tokenizer=tokenizer
            )
            
            # 获取LLaVA模型的投影层和隐藏层大小
            self.vision_projector = llava_model.get_model().mm_projector
            self.hidden_size = llava_model.config.hidden_size
            
            # 确保特征融合模块和投影层在同一设备上
            device = next(llava_model.parameters()).device
            self.to(device)
            
            self.initialized = True
        
        return self
        
    def forward(self, image):
        """前向传播，融合特征并通过投影层"""
        if not self.initialized:
            raise RuntimeError("FusedVisionProjector尚未初始化，请先调用initialize方法")
        
        self.debug_step += 1
        
        # 获取融合特征
        fusion_result = self.feature_fusion.forward(image)
        fused_features = fusion_result['fused_features']
        
        # 再次检查融合特征中的nan值
        if torch.isnan(fused_features).any():
            print(f"投影器步骤 {self.debug_step} - 警告: 进入投影层前的融合特征中存在nan值")
            fused_features = torch.nan_to_num(fused_features, nan=1e-8)
        
        # 通过投影层
        # 这里需要注意，投影层的输入形状需要匹配
        # 对于LLaVA的投影层，输入通常是 [batch_size, num_patches, vision_hidden_size]
        # 输出是 [batch_size, num_patches, llm_hidden_size]
        
        # 我们需要确保fused_features的最后一个维度与投影层的输入维度匹配
        # 如果不匹配，可能需要添加一个额外的投影层
        try:
            # 首先检查维度是否匹配
            if fused_features.shape[-1] != (self.vision_projector[0].in_features if isinstance(self.vision_projector, nn.Sequential) else self.vision_projector.in_features):
                # 创建一个稳定的适配器层来调整维度
                input_dim = fused_features.shape[-1]
                output_dim = self.vision_projector[0].in_features if isinstance(self.vision_projector, nn.Sequential) else self.vision_projector.in_features
                
                # 确保适配器与特征在同一设备和数据类型
                device = fused_features.device
                dtype = fused_features.dtype
                
                adapter = nn.Linear(input_dim, output_dim, device=device, dtype=dtype)
                nn.init.xavier_uniform_(adapter.weight)
                if adapter.bias is not None:
                    nn.init.zeros_(adapter.bias)
                
                # 应用适配器
                fused_features = adapter(fused_features)
                
                # 检查适配器输出的nan值
                if torch.isnan(fused_features).any():
                    print(f"投影器步骤 {self.debug_step} - 警告: 适配器输出中存在nan值")
                    fused_features = torch.nan_to_num(fused_features, nan=1e-8)
                
            # 通过投影层
            try:
                # 通过投影层
                projected_features = self.vision_projector(fused_features)
                
                # 检查投影后的nan值
                if torch.isnan(projected_features).any():
                    print(f"投影器步骤 {self.debug_step} - 警告: 投影后的特征中存在nan值")
                    projected_features = torch.nan_to_num(projected_features, nan=1e-8)
            except Exception as e:
                # 尝试使用更稳定的投影方法
                with torch.autocast(device_type='cuda', enabled=False):
                    projected_features = self.vision_projector(fused_features.float())
                    if torch.isnan(projected_features).any():
                        print(f"投影器步骤 {self.debug_step} - 警告: 替代投影方法后仍存在nan值")
                        projected_features = torch.nan_to_num(projected_features, nan=1e-8)
        except RuntimeError as e:
            # 尝试使用更简单的处理方式
            if len(fused_features.shape) == 3:
                # 对于[batch, seq_len, hidden_dim]格式的数据，尝试使用平均池化降低维度
                from torch.nn import AdaptiveAvgPool1d
                pooler = AdaptiveAvgPool1d(1024).to(fused_features.device)
                # 需要调整形状以适应池化层
                batch_size, seq_len, hidden_dim = fused_features.shape
                fused_features_reshaped = fused_features.transpose(1, 2)  # [batch, hidden_dim, seq_len]
                
                # 添加数值稳定性保障
                if torch.isnan(fused_features_reshaped).any():
                    print(f"投影器步骤 {self.debug_step} - 警告: 池化前特征中存在nan值")
                    fused_features_reshaped = torch.nan_to_num(fused_features_reshaped, nan=1e-8)
                
                pooled_features = pooler(fused_features_reshaped)  # [batch, 1024, seq_len]
                fused_features = pooled_features.transpose(1, 2)  # [batch, seq_len, 1024]
                
                # 再次检查nan值
                if torch.isnan(fused_features).any():
                    print(f"投影器步骤 {self.debug_step} - 警告: 池化后特征中存在nan值")
                    fused_features = torch.nan_to_num(fused_features, nan=1e-8)
                
                # 再次尝试通过投影层
                try:
                    projected_features = self.vision_projector(fused_features)
                    
                    # 检查投影后的nan值
                    if torch.isnan(projected_features).any():
                        print(f"投影器步骤 {self.debug_step} - 警告: 池化后投影特征中存在nan值")
                        projected_features = torch.nan_to_num(projected_features, nan=1e-8)
                except Exception as e2:
                    # 如果所有方法都失败，返回零向量以避免训练中断
                    batch_size, *rest = fused_features.shape
                    hidden_dim = self.hidden_size
                    projected_features = torch.zeros((batch_size,) + tuple(rest[:-1]) + (hidden_dim,), 
                                                  device=fused_features.device, 
                                                  dtype=fused_features.dtype)
            else:
                # 如果所有方法都失败，返回零向量以避免训练中断
                batch_size, *rest = fused_features.shape
                hidden_dim = self.hidden_size
                projected_features = torch.zeros((batch_size,) + tuple(rest[:-1]) + (hidden_dim,), 
                                              device=fused_features.device, 
                                              dtype=fused_features.dtype)
        
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