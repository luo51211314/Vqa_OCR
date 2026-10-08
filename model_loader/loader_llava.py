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
        """根据指标类型获取推理配置（无随机性版本）"""
        configs = {
            "anls": {
                # "temperature": 0.1,
                # "top_p": 0.7,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 64,  # 短输出
                "num_beams": 4,
                "prompt_suffix": "\nPlease provide a concise answer directly without any explanation. If the question involves numerical values, please output in Arabic numeral form (e.g., 1, 2, 3) instead of English words (one, two, three)."
            },
            "relaxed_accuracy": {
                # "temperature": 0.8,
                # "top_p": 0.9,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 256,
                "num_beams": 1,
                "prompt_suffix": "\nPlease provide a concise answer directly. If the question involves numerical values, please output in Arabic numeral form (e.g., 1, 2, 3) instead of English words (one, two, three)."
            },
            "relaxed_accuracy_80": {
                # "temperature": 0.8,
                # "top_p": 0.9,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 256,
                "num_beams": 1,
                "prompt_suffix": "\nPlease provide a concise answer directly. If the question involves numerical values, please output in Arabic numeral form (e.g., 1, 2, 3) instead of English words (one, two, three)."
            },
            "bleu": {
                # "temperature": 0.7,
                # "top_p": 0.9,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 256,  # 长输出
                "num_beams": 1,
                "prompt_suffix": "\nPlease provide a complete and detailed answer, including all relevant information. If the question involves numerical values, please output in Arabic numeral form (e.g., 1, 2, 3) instead of English words (one, two, three). Please answer in the same sentence structure as the question: for yes/no questions, answer with 'Yes' or 'No'; for counting questions, answer with the number; for comparison questions, state the specific values and calculate the difference."
            },
            "weighted": {
                # "temperature": 0.7,
                # "top_p": 0.9,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 256,  # 控制输出长度在50词左右
                "num_beams": 1,
                "prompt_suffix": "\nPlease provide a complete but concise answer (within 100 words). Include all relevant information, but not lengthy, and your answer should be 1-3 complete sentences. If the question involves numerical values, please output in Arabic numeral form (e.g., 1, 2, 3) instead of English words."
            }
        }
        if metric_type not in configs:
            raise ValueError(f"不支持的metric_type: {metric_type}")
        return configs[metric_type]
    
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