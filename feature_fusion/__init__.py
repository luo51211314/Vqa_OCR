"""
特征融合模块
包含对比学习特征对齐和注意力机制特征融合
"""

from .contrastive_alignment import ContrastiveAlignment
from .feature_aligner import FeatureAligner
from .attention_fusion import AttentionFusion
from .fusion_manager import FusionManager

__all__ = [
    'ContrastiveAlignment',
    'FeatureAligner', 
    'AttentionFusion',
    'FusionManager'
]