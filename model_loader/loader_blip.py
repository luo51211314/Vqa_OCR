import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from .loader_base import BaseModelLoader


class BlipLoader(BaseModelLoader):
    """InstructBLIP 模型加载器"""

    def __init__(self):
        super().__init__()
        self.processor = None
        self.max_image_size = 1024

    def load_model(self, model_path, device="cuda"):
        """加载 InstructBLIP 模型"""
        # 使用 use_fast=False 避免 tokenizer 兼容性问题
        processor = InstructBlipProcessor.from_pretrained(model_path, use_fast=False)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )
        self.model = model
        self.processor = processor
        return None, model, processor, None

    def get_inference_config(self, metric_type):
        """根据指标类型获取推理配置（无随机性版本）"""
        configs = {
            "anls": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 64,
                "repetition_penalty": 1.1,
                "prompt_suffix": " Answer:"
            },
            "relaxed_accuracy": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 64,
                "repetition_penalty": 1.1,
                "prompt_suffix": " Answer:"
            },
            "relaxed_accuracy_80": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 64,
                "repetition_penalty": 1.1,
                "prompt_suffix": " Answer:"
            },
            "bleu": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 64,
                "repetition_penalty": 1.1,
                "prompt_suffix": " Answer:"
            },
            "weighted": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": 64,
                "repetition_penalty": 1.1,
                "prompt_suffix": " Answer:"
            }
        }
        return configs.get(metric_type, configs["anls"])

    def process_prompt(self, prompt, metric_type):
        """根据指标类型处理 prompt

        InstructBLIP 使用格式: Question: {question} Answer:
        """
        # InstructBLIP 使用特定的 prompt 格式
        formatted_prompt = f"Question: {prompt} Answer:"
        return formatted_prompt

    def generate(self, images, prompts, config):
        """生成回答

        Args:
            images: 图像列表 (PIL Images)
            prompts: prompt 列表
            config: 推理配置

        Returns:
            生成的文本列表
        """
        if not isinstance(images, list):
            images = [images]
        if not isinstance(prompts, list):
            prompts = [prompts]

        results = []
        for image, prompt in zip(images, prompts):
            # 准备输入
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)

            # 生成答案
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=config.get("do_sample", True),
                    temperature=config.get("temperature", 0.1),
                    top_p=config.get("top_p", 0.9),
                    max_new_tokens=config.get("max_new_tokens", 64),
                    repetition_penalty=config.get("repetition_penalty", 1.1),
                )

            # 解码答案
            answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            results.append(answer)

        return results

    def decode(self, output_ids, tokenizer):
        """解码输出 - 这里我们不会使用这个方法"""
        raise NotImplementedError("BlipLoader.decode is not used. Use processor.batch_decode.")

    def tokenizer_image_token(self, prompt, tokenizer, image_processor, return_tensors="pt"):
        """InstructBLIP 不使用这个方法，它使用 processor 处理"""
        raise NotImplementedError("BlipLoader.tokenizer_image_token is not used.")
