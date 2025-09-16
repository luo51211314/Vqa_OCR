from abc import ABC, abstractmethod
import torch

class BaseModelLoader(ABC):
    """模型加载器基类"""
    
    @abstractmethod
    def load_model(self, model_path, device="cuda"):
        """加载模型"""
        pass
    
    @abstractmethod
    def get_inference_config(self, metric_type):
        """根据指标类型获取推理配置"""
        pass
    
    @abstractmethod
    def process_prompt(self, prompt, metric_type):
        """根据指标类型处理prompt"""
        pass
    
    @abstractmethod
    def generate(self, input_ids, images, attention_mask, image_sizes, config):
        """生成回答"""
        pass
    
    @abstractmethod
    def decode(self, output_ids, tokenizer):
        """解码输出"""
        pass
    
    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=None, return_tensors=None):
        """
        处理包含图像token的prompt
        默认实现：使用标准tokenizer
        """
        return tokenizer(prompt, return_tensors=return_tensors).input_ids