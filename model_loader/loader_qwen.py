import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .loader_base import BaseModelLoader

class QwenLoader(BaseModelLoader):
    """Qwen模型加载器"""
    
    def load_model(self, model_path, device="cuda"):
        """加载Qwen模型"""
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model = model
        self.tokenizer = tokenizer
        return tokenizer, model, None, None
    
    def get_inference_config(self, metric_type):
        """根据指标类型获取推理配置"""
        configs = {
            "anls": {
                "temperature": 0.1,
                "top_p": 0.7,
                "max_new_tokens": 64,
                "num_beams": 4,
                "prompt_suffix": "\n请直接给出简洁答案，不要解释。"
            },
            "relaxed_accuracy": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 256,
                "num_beams": 1,
                "prompt_suffix": "\n请提供完整详细的答案，包含所有相关信息。"
            },
            "relaxed_accuracy_80": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 256,
                "num_beams": 1,
                "prompt_suffix": "\n请提供完整详细的答案，包含所有相关信息。"
            }
        }
        return configs.get(metric_type, configs["anls"])
    
    def process_prompt(self, prompt, metric_type):
        """根据指标类型处理prompt"""
        config = self.get_inference_config(metric_type)
        prompt_suffix = config.get("prompt_suffix", "")
        
        # Qwen的prompt格式
        return f"<image>\n{prompt}{prompt_suffix}"
    
    def generate(self, input_ids, images, attention_mask, image_sizes, config):
        """生成回答"""
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True if config["temperature"] > 0 else False,
                temperature=config["temperature"],
                top_p=config["top_p"],
                num_beams=config["num_beams"],
                max_new_tokens=config["max_new_tokens"],
                use_cache=True,
            )
        return output_ids
    
    def decode(self, output_ids, tokenizer):
        """解码输出"""
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()