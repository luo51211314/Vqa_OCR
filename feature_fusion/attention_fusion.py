import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AttentionFusion(nn.Module):
    """
    注意力机制特征融合模块
    使用注意力机制融合对齐后的LLaVA特征和专家特征
    """
    
    def __init__(self, 
                 llava_hidden_size: int = 4096,
                 expert_hidden_size: int = 768,
                 projection_dim: int = 512,
                 num_heads: int = 8):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        
        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(projection_dim)
        self.layer_norm2 = nn.LayerNorm(projection_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 4),
            nn.ReLU(),
            nn.Linear(projection_dim * 4, projection_dim)
        )
        
        # 残差连接
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                aligned_llava: torch.Tensor,
                aligned_expert: torch.Tensor) -> torch.Tensor:
        """
        注意力特征融合
        
        Args:
            aligned_llava: 对齐后的LLaVA特征 [batch_size, seq_len, projection_dim]
            aligned_expert: 对齐后的专家特征 [batch_size, seq_len, projection_dim]
        
        Returns:
            fused_features: 融合后的特征 [batch_size, seq_len, projection_dim]
        """
        batch_size, seq_len, _ = aligned_llava.shape
        
        # 确保输入数据类型一致
        if aligned_llava.dtype != aligned_expert.dtype:
            aligned_expert = aligned_expert.to(aligned_llava.dtype)
        
        # 将专家特征作为查询，LLaVA特征作为键和值
        query = aligned_expert
        key = aligned_llava
        value = aligned_llava
        
        # 多头注意力
        attended_features, attention_weights = self.multihead_attention(
            query, key, value
        )
        
        # 残差连接和层归一化
        attended_features = self.layer_norm1(aligned_expert + self.dropout(attended_features))
        
        # 前馈网络
        ff_output = self.feed_forward(attended_features)
        
        # 最终输出
        fused_features = self.layer_norm2(attended_features + self.dropout(ff_output))
        
        return fused_features