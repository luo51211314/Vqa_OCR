import torch
import torch.nn as nn
from typing import Tuple

class FeatureAligner(nn.Module):
    """
    特征维度对齐模块
    将不同维度的特征投影到相同的维度空间
    """
    
    def __init__(self, 
                 llava_hidden_size: int = 1024,  # 修正：实际LLaVA特征隐藏大小为1024
                 expert_hidden_size: int = 1024,  # 修正：实际专家特征隐藏大小也为1024
                 projection_dim: int = 512):
        super().__init__()
        
        self.llava_hidden_size = llava_hidden_size
        self.expert_hidden_size = expert_hidden_size
        self.projection_dim = projection_dim
        
        # LLaVA特征投影
        self.llava_projection = nn.Sequential(
            nn.Linear(llava_hidden_size, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # 专家特征投影
        self.expert_projection = nn.Sequential(
            nn.Linear(expert_hidden_size, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, 
                llava_features: torch.Tensor,
                expert_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        特征维度对齐
        
        Args:
            llava_features: [batch_size, seq_len, llava_hidden_size] or [seq_len, llava_hidden_size]
            expert_features: [batch_size, seq_len, expert_hidden_size] or [seq_len, expert_hidden_size]
        
        Returns:
            aligned_llava: [batch_size, seq_len, projection_dim]
            aligned_expert: [batch_size, seq_len, projection_dim]
        """
        # 确保输入数据类型一致
        if llava_features.dtype != expert_features.dtype:
            expert_features = expert_features.to(llava_features.dtype)
        
        # 保存输入数据类型用于投影层
        self.input_dtype = llava_features.dtype
        
        # 处理不同的输入维度
        if len(llava_features.shape) == 2:
            # [seq_len, hidden_size] -> [1, seq_len, hidden_size]
            llava_features = llava_features.unsqueeze(0)
            expert_features = expert_features.unsqueeze(0)
        
        batch_size, llava_seq_len, actual_llava_hidden = llava_features.shape
        _, expert_seq_len, actual_expert_hidden = expert_features.shape
        
        # 检查隐藏大小是否匹配
        if actual_llava_hidden != self.llava_hidden_size:
            # 动态调整投影层以适应实际输入维度
            self._adjust_projection_layers(actual_llava_hidden, actual_expert_hidden)
        
        # 使用实际的特征维度进行重塑，而不是模块的预设值
        llava_flat = llava_features.reshape(-1, actual_llava_hidden)
        expert_flat = expert_features.reshape(-1, actual_expert_hidden)
        
        # 投影到相同维度
        aligned_llava_flat = self.llava_projection(llava_flat)
        aligned_expert_flat = self.expert_projection(expert_flat)
        
        # 恢复原始形状
        aligned_llava = aligned_llava_flat.reshape(batch_size, llava_seq_len, self.projection_dim)
        aligned_expert = aligned_expert_flat.reshape(batch_size, expert_seq_len, self.projection_dim)
        
        return aligned_llava, aligned_expert
    
    def _adjust_projection_layers(self, actual_llava_hidden: int, actual_expert_hidden: int):
        """动态调整投影层以适应实际输入维度"""
        device = next(self.parameters()).device
        
        # 重新初始化LLaVA投影层
        self.llava_projection = nn.Sequential(
            nn.Linear(actual_llava_hidden, self.projection_dim * 2),
            nn.ReLU(),
            nn.Linear(self.projection_dim * 2, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        ).to(device)
        
        # 重新初始化专家投影层
        self.expert_projection = nn.Sequential(
            nn.Linear(actual_expert_hidden, self.projection_dim * 2),
            nn.ReLU(),
            nn.Linear(self.projection_dim * 2, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        ).to(device)
        
        # 确保投影层权重与输入数据类型一致
        if hasattr(self, 'input_dtype'):
            self.llava_projection = self.llava_projection.to(self.input_dtype)
            self.expert_projection = self.expert_projection.to(self.input_dtype)
        
        # 更新模块参数
        self.llava_hidden_size = actual_llava_hidden
        self.expert_hidden_size = actual_expert_hidden
        
        print(f"投影层已动态调整: LLaVA隐藏大小={actual_llava_hidden}, 专家隐藏大小={actual_expert_hidden}")