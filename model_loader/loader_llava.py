import sys
import os
import torch
from models.llava.llava.model.builder import load_pretrained_model
from models.llava.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from models.llava.llava.conversation import conv_templates
from models.llava.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from .loader_base import BaseModelLoader

class LLaVALoader(BaseModelLoader):
    """LLaVA模型加载器"""
    
    def load_model(self, model_path, device="cuda"):
        """加载LLaVA模型"""
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device=device,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
        self.conv_mode = "llava_v1"
        return tokenizer, model, image_processor, context_len
    
    def get_inference_config(self, metric_type):
        """根据指标类型获取推理配置"""
        configs = {
            "anls": {
                "temperature": 0.1,
                "top_p": 0.7,
                "max_new_tokens": 64,  # 短输出
                "num_beams": 4,
                "prompt_suffix": "\n请直接给出简洁答案，不要解释。"
            },
            "relaxed_accuracy": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 256,  # 长输出
                "num_beams": 1,
                "prompt_suffix": "\n请提供完整详细的答案，包含所有相关信息。"
            },
            "relaxed_accuracy_80": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 256,  # 长输出
                "num_beams": 1,
                "prompt_suffix": "\n请提供完整详细的答案，包含所有相关信息。"
            }
        }
        return configs.get(metric_type, configs["anls"])
    
    def process_prompt(self, prompt, metric_type):
        """根据指标类型处理prompt"""
        config = self.get_inference_config(metric_type)
        prompt_suffix = config.get("prompt_suffix", "")
        
        use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if use_im_start_end
            else DEFAULT_IMAGE_TOKEN
        )
        
        qs = image_token_se + "\n" + prompt + prompt_suffix
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def generate(self, input_ids, images, attention_mask, image_sizes, config):
        """生成回答"""
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
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
    
    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=None, return_tensors=None):
        """处理包含图像token的prompt"""
        from models.llava.llava.mm_utils import tokenizer_image_token as llava_tokenizer_image_token
        from models.llava.llava.constants import IMAGE_TOKEN_INDEX
        
        if image_token_index is None:
            image_token_index = IMAGE_TOKEN_INDEX
        
        return llava_tokenizer_image_token(
            prompt, tokenizer, image_token_index, return_tensors
        )