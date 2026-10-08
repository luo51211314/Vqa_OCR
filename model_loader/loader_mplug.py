import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from .loader_base import BaseModelLoader

class MplugLoader(BaseModelLoader):
    """mPLUG-Owl3 模型加载器"""
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def load_model(self, model_path, device="cuda"):
        """加载 mPLUG-Owl3 模型"""
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )
        model = model.to(device)
        model.eval()
        
        # 初始化 processor
        processor = model.init_processor(tokenizer)
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        
        return tokenizer, model, processor, None
    
    def get_inference_config(self, metric_type):
        """根据指标类型获取推理配置"""
        configs = {
            "anls": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 64,
                "prompt_suffix": "\nAnswer briefly."
            },
            "relaxed_accuracy": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 128,
                "prompt_suffix": "\nAnswer briefly. If digits in answer, use Arabic numerals."
            },
            "relaxed_accuracy_80": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 128,
                "prompt_suffix": "\nAnswer briefly. If digits in answer, use Arabic numerals."
            },
            "bleu": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 128,
                "prompt_suffix": "\nAnswer in 1-3 sentences. If digits in answer, use Arabic numerals."
            },
            "weighted": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 128,
                "prompt_suffix": "\nAnswer in 1-3 sentences. If digits in answer, use Arabic numerals."
            }
        }
        return configs.get(metric_type, configs["anls"])
    
    def process_prompt(self, prompt, metric_type):
        """根据指标类型处理 prompt"""
        config = self.get_inference_config(metric_type)
        prompt_suffix = config.get("prompt_suffix", "")
        return prompt + prompt_suffix
    
    def generate(self, image, prompt, max_new_tokens, **kwargs):
        """生成回答"""
        IMAGE_TOKEN = "<|image|>"
        messages = [
            {"role": "user", "content": IMAGE_TOKEN + "\n" + prompt},
            {"role": "assistant", "content": ""}
        ]
        
        # 处理输入
        inputs = self.processor(messages, images=[image], videos=None)
        inputs = inputs.to(self.model.device)
        
        # 生成
        with torch.no_grad():
            answer = self.model.generate(
                **inputs,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                decode_text=True,
            )
        
        return answer[0] if isinstance(answer, list) else answer
    
    def decode(self, output_ids, tokenizer):
        """解码输出 - mPLUG 使用自己的 generate 方法返回文本"""
        raise NotImplementedError("MplugLoader.decode is not used. Use generate method directly.")
    
    def tokenizer_image_token(self, prompt, tokenizer, image_processor, return_tensors="pt"):
        """mPLUG 不使用这个方法，它使用 processor 处理"""
        raise NotImplementedError("MplugLoader.tokenizer_image_token is not used.")
