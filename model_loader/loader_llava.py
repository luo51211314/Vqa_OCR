import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 首先添加 llava 目录到 sys.path，以便 builder.py 内部的 from llava.model 能正常工作
llava_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "llava")
if llava_dir not in sys.path:
    sys.path.insert(0, llava_dir)

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
        """统一的确定性推理配置（三模型共用 common_suffix，禁止模型私有后缀）。"""
        from .common_suffix import get_generation_config, get_prompt_suffix
        config = get_generation_config(metric_type)
        config["prompt_suffix"] = get_prompt_suffix(metric_type)
        return config
    
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