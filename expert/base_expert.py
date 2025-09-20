from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

class BaseExpert(ABC):
    """专家模块基类"""
    
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.model = None
        self.initialized = False
    
    @abstractmethod
    def initialize(self, model_path: Optional[str] = None, **kwargs):
        """初始化专家模型"""
        pass
    
    @abstractmethod
    def process(self, image, question: Optional[str] = None) -> Dict[str, Any]:
        """处理输入图像，返回结构化信息"""
        pass
    
    @abstractmethod
    def to_prompt(self, result: Dict[str, Any]) -> str:
        """将处理结果转换为LLM可理解的提示词"""
        pass
    
    def is_available(self) -> bool:
        """检查专家模块是否可用"""
        return self.initialized
    
    def get_info(self) -> Dict[str, Any]:
        """获取专家模块信息"""
        return {
            "name": self.name,
            "device": self.device,
            "initialized": self.initialized
        }