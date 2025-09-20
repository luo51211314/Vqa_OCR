import importlib
import os
from typing import Dict, List, Any, Optional
from .base_expert import BaseExpert

class ExpertManager:
    """专家模块管理器"""
    
    def __init__(self):
        self.experts: Dict[str, BaseExpert] = {}
        self.available_experts = self._discover_experts()
    
    def _discover_experts(self) -> List[str]:
        """自动发现可用的专家模块"""
        expert_dir = os.path.dirname(__file__)
        experts = []
        
        for file in os.listdir(expert_dir):
            if file.endswith("_expert.py") and file != "base_expert.py":
                expert_name = file[:-10]  # 移除'_expert.py'
                experts.append(expert_name)
        
        return experts
    
    def register_expert(self, name: str, expert: BaseExpert):
        """注册专家模块"""
        self.experts[name] = expert
    
    def get_expert(self, name: str) -> Optional[BaseExpert]:
        """获取专家模块实例"""
        return self.experts.get(name)
    
    def initialize_expert(self, name: str, model_path: Optional[str] = None, **kwargs):
        """初始化特定专家模块"""
        if name not in self.available_experts:
            raise ValueError(f"专家模块 {name} 不可用，可用模块: {self.available_experts}")
        
        # 动态导入专家模块
        module_name = f"expert.{name}_expert"
        try:
            module = importlib.import_module(module_name)
            expert_class = getattr(module, f"{name.capitalize()}Expert")
            expert_instance = expert_class()
            expert_instance.initialize(model_path, **kwargs)
            self.register_expert(name, expert_instance)
            return expert_instance
        except ImportError as e:
            raise ImportError(f"无法导入专家模块 {name}: {e}")
    
    def process_with_experts(self, image, question: str, expert_names: List[str]) -> str:
        """使用多个专家模块处理输入"""
        expert_outputs = []
        
        for expert_name in expert_names:
            expert = self.get_expert(expert_name)
            if expert and expert.is_available():
                try:
                    result = expert.process(image, question)
                    prompt_text = expert.to_prompt(result)
                    expert_outputs.append(prompt_text)
                except Exception as e:
                    print(f"专家模块 {expert_name} 处理失败: {e}")
        
        # 组合所有专家输出
        combined_prompt = "\n".join(expert_outputs)
        final_prompt = f"{combined_prompt}\n\n基于以上信息，请回答以下问题: {question}"
        
        return final_prompt
    
    def get_available_experts(self) -> List[str]:
        """获取所有可用专家模块"""
        return list(self.experts.keys())