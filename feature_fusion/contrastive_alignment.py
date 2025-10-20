import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import os

# 导入FeatureAligner
try:
    from .feature_aligner import FeatureAligner
except ImportError:
    # 直接运行脚本时的导入方式
    from feature_aligner import FeatureAligner

class ContrastiveAlignment(nn.Module):
    """
    对比学习对齐模块（重命名，不涉及融合，只有对齐）
    1. 特征维度对齐
    2. 对比学习特征对齐
    """
    
    def __init__(self, 
                 llava_hidden_size: int = 1024,  # 实际LLaVA特征隐藏大小为1024
                 expert_hidden_size: int = 1024,  # 修正：实际专家特征隐藏大小也为1024
                 projection_dim: int = 512,
                 temperature: float = 0.1):  # 降低温度参数以增加梯度稳定性
        super().__init__()
        
        self.llava_hidden_size = llava_hidden_size
        self.expert_hidden_size = expert_hidden_size
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # 特征对齐器
        self.feature_aligner = FeatureAligner(
            llava_hidden_size=llava_hidden_size,
            expert_hidden_size=expert_hidden_size,
            projection_dim=projection_dim
        )
        
        # 对比学习投影头
        self.contrastive_projection = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 添加权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重以提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                llava_features: torch.Tensor,
                expert_features: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mode: str = "train") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对比学习对齐
        
        Args:
            llava_features: LLaVA特征 [batch_size, seq_len, hidden_size]
            expert_features: 专家模型特征 [batch_size, seq_len, expert_hidden_size]
            labels: 用于对比学习的标签（可选）
            mode: 模式 ("train", "inference")
        
        Returns:
            aligned_llava: 对齐后的LLaVA特征
            aligned_expert: 对齐后的专家特征
            contrastive_loss: 对比学习损失
        """
        
        # 1. 特征维度对齐
        aligned_llava, aligned_expert = self.feature_aligner(llava_features, expert_features)
        
        # 2. 对比学习（仅在训练模式下）
        contrastive_loss = None
        if mode == "train":
            contrastive_loss = self._contrastive_learning(aligned_llava, aligned_expert)
        
        return aligned_llava, aligned_expert, contrastive_loss
    
    def _contrastive_learning(self, 
                            aligned_llava: torch.Tensor,
                            aligned_expert: torch.Tensor) -> torch.Tensor:
        """
        对比学习损失计算
        同一图片的LLaVA特征和专家特征为正样本对
        不同图片的特征为负样本对
        """
        batch_size = aligned_llava.size(0)
        
        # 投影到对比学习空间
        llava_proj = self.contrastive_projection(aligned_llava.mean(dim=1))  # [batch_size, proj_dim]
        expert_proj = self.contrastive_projection(aligned_expert.mean(dim=1))  # [batch_size, proj_dim]
        
        # 归一化
        llava_proj = F.normalize(llava_proj, p=2, dim=1)
        expert_proj = F.normalize(expert_proj, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(llava_proj, expert_proj.t()) / self.temperature
        
        # 创建标签（对角线元素是正样本）
        labels = torch.arange(batch_size, device=aligned_llava.device)
        
        # 计算对比损失（InfoNCE损失）
        # 使用cross_entropy实现更稳定的计算
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        return contrastive_loss
    
    def save_weights(self, filename: str):
        """保存融合模块权重"""
        weight_path = os.path.join(self.weight_dir, filename)
        torch.save(self.state_dict(), weight_path)
        print(f"融合模块权重已保存到: {weight_path}")
    
    def load_weights(self, filename: str):
        """加载融合模块权重"""
        weight_path = os.path.join(self.weight_dir, filename)
        if os.path.exists(weight_path):
            self.load_state_dict(torch.load(weight_path))
            print(f"融合模块权重已从 {weight_path} 加载")
        else:
            print(f"警告: 权重文件 {weight_path} 不存在，使用随机初始化")
    
    def freeze_llava_parameters(self, llava_model):
        """冻结LLaVA模型参数"""
        for param in llava_model.parameters():
            param.requires_grad = False
        print("LLaVA模型参数已冻结")
    
    def get_trainable_parameters(self):
        """获取可训练参数（仅融合模块）"""
        return self.parameters()