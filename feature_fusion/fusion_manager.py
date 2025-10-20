import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os
from .contrastive_alignment import ContrastiveAlignment

class FusionManager:
    """
    特征融合管理器
    负责管理对比学习融合模块的初始化和推理
    """
    
    def __init__(self):
        self.fusion_module = None
        self.expert_extractor = None
    
    def initialize_fusion(self, 
                         fusion_type: str,
                         weight_path: Optional[str] = None,
                         device: str = "cuda") -> bool:
        """
        初始化融合模块
        
        Args:
            fusion_type: 融合类型 ("paddleocr", "pix2struct")
            weight_path: 权重文件路径
            device: 设备
        
        Returns:
            success: 是否初始化成功
        """
        try:
            # 根据融合类型设置专家特征维度
            if fusion_type == "paddleocr":
                expert_hidden_size = 768  # PaddleOCR特征维度
            elif fusion_type == "pix2struct":
                expert_hidden_size = 768  # Pix2Struct特征维度
            else:
                raise ValueError(f"不支持的融合类型: {fusion_type}")
            
            # 初始化融合模块（与训练时保存的权重维度保持一致）
            self.fusion_module = ContrastiveAlignment(
                llava_hidden_size=1024,  # 输入维度：实际LLaVA特征隐藏大小为1024
                expert_hidden_size=expert_hidden_size,
                projection_dim=512,
                fusion_method="attention",
                output_dim=4096  # 输出维度：与训练权重保持一致
            )
            
            # 加载权重
            if weight_path and os.path.exists(weight_path):
                self.fusion_module.load_weights(weight_path)
                print(f"融合模块权重已加载: {weight_path}")
            else:
                # 尝试加载默认权重
                default_weight = f"/root/autodl-tmp/weight/fusion_module_final_{fusion_type}.pth"
                if os.path.exists(default_weight):
                    self.fusion_module.load_weights(default_weight)
                    print(f"使用默认权重: {default_weight}")
                else:
                    print("警告: 未找到融合模块权重，使用随机初始化")
            
            self.fusion_module.to(device)
            self.fusion_module.eval()
            
            # 初始化专家特征提取器
            self._initialize_expert_extractor(fusion_type, device)
            
            print(f"对比学习融合模块初始化成功: {fusion_type}")
            return True
            
        except Exception as e:
            print(f"融合模块初始化失败: {e}")
            self.fusion_module = None
            return False
    
    def _initialize_expert_extractor(self, fusion_type: str, device: str):
        """初始化专家特征提取器"""
        try:
            if fusion_type == "paddleocr":
                from .real_feature_extractor import RealFeatureExtractor
                self.expert_extractor = RealFeatureExtractor(model_type="paddleocr", device=device)
            elif fusion_type == "pix2struct":
                from .real_feature_extractor import RealFeatureExtractor
                self.expert_extractor = RealFeatureExtractor(model_type="pix2struct", device=device)
            else:
                raise ValueError(f"不支持的专家类型: {fusion_type}")
                
            print(f"专家特征提取器初始化成功: {fusion_type}")
        except Exception as e:
            print(f"专家特征提取器初始化失败: {e}")
            self.expert_extractor = None
    
    def extract_expert_features(self, image, question="") -> Optional[torch.Tensor]:
        """提取专家特征"""
        if self.expert_extractor is None:
            return None
        
        try:
            return self.expert_extractor.extract_features(image, question)
        except Exception as e:
            print(f"专家特征提取失败: {e}")
            return None
    
    def fuse_features(self, 
                     llava_features: torch.Tensor,
                     expert_features: torch.Tensor) -> Optional[torch.Tensor]:
        """特征融合"""
        if self.fusion_module is None:
            return None
        
        try:
            # 强制将所有特征转换为float32，避免数据类型不匹配问题
            print(f"融合前数据类型 - LLaVA: {llava_features.dtype}, 专家: {expert_features.dtype}")
            llava_features = llava_features.to(torch.float32)
            expert_features = expert_features.to(torch.float32)
            print(f"数据类型转换后 - LLaVA: {llava_features.dtype}, 专家: {expert_features.dtype}")
            
            with torch.no_grad():
                fused_features, _ = self.fusion_module(
                    llava_features, expert_features, mode="inference"
                )
            
            # 将融合后的特征转换回原始数据类型
            if llava_features.dtype != fused_features.dtype:
                fused_features = fused_features.to(llava_features.dtype)
            
            return fused_features
        except Exception as e:
            print(f"特征融合失败: {e}")
            return None
    
    def is_initialized(self) -> bool:
        """检查融合模块是否已初始化"""
        return self.fusion_module is not None