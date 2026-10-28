import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .loader_base import BaseModelLoader

class QwenLoader(BaseModelLoader):
    """Qwen2.5-VL模型加载器"""
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.max_image_size = 1024
    
    def load_model(self, model_path, device="cuda"):
        """加载Qwen2.5-VL模型"""
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.model = model
        self.processor = processor
        return None, model, processor, None
    
    def get_inference_config(self, metric_type):
        """统一的确定性推理配置（三模型共用 common_suffix，禁止模型私有后缀）。"""
        from .common_suffix import get_generation_config, get_prompt_suffix
        config = get_generation_config(metric_type)
        config["prompt_suffix"] = get_prompt_suffix(metric_type)
        return config
    
    def process_prompt(self, prompt, metric_type):
        """根据指标类型处理prompt"""
        config = self.get_inference_config(metric_type)
        prompt_suffix = config.get("prompt_suffix", "")
        return prompt + prompt_suffix
    
    def generate(self, input_ids, image_tensor, attention_mask, image_sizes, config):
        """生成回答 - 这里我们不会使用这个方法，因为Qwen2.5-VL有自己的生成方式"""
        raise NotImplementedError("QwenLoader.generate is not used. Use direct model.generate with processor inputs.")
    
    def decode(self, output_ids, tokenizer):
        """解码输出 - 这里我们不会使用这个方法"""
        raise NotImplementedError("QwenLoader.decode is not used. Use processor.batch_decode.")
    
    def tokenizer_image_token(self, prompt, tokenizer, image_processor, return_tensors="pt"):
        """Qwen2.5-VL不使用这个方法，它使用processor处理"""
        raise NotImplementedError("QwenLoader.tokenizer_image_token is not used.")
