import torch
import os
from typing import Dict, Any, Optional
from .base_expert import BaseExpert
from PIL import Image
import numpy as np

class Pix2structExpert(BaseExpert):
    """Pix2Struct图像转HTML专家模块"""
    
    def __init__(self):
        super().__init__("pix2struct")
        self.model = None
        self.processor = None
    
    def initialize(self, model_path: Optional[str] = None, **kwargs):
        """初始化Pix2Struct模型"""
        try:
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
            
            # 使用微调后的模型
            model_dir = model_path or "/root/autodl-tmp/weight/pix2struct_finetuned"
            
            # 加载处理器和模型
            try:
                # 尝试使用兼容的方式加载processor，解决不同transformers版本的兼容性问题
                from transformers import T5Tokenizer
                
                # 先加载tokenizer，使用更兼容的方式
                tokenizer = T5Tokenizer.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    use_fast=False,
                    # 禁用可能有问题的pre_tokenizer配置
                    legacy=False
                )
                
                # 然后加载image processor
                from transformers import Pix2StructImageProcessor
                image_processor = Pix2StructImageProcessor.from_pretrained(
                    model_dir,
                    local_files_only=True
                )
                
                # 创建processor
                self.processor = Pix2StructProcessor(
                    image_processor=image_processor,
                    tokenizer=tokenizer
                )
            except Exception as e1:
                print(f"第一种加载方式失败: {str(e1)}")
                try:
                    # 如果上述方法失败，尝试直接加载processor但添加兼容参数
                    self.processor = Pix2StructProcessor.from_pretrained(
                        model_dir,
                        local_files_only=True,
                        trust_remote_code=False,
                        tokenizer_class="T5Tokenizer",
                        use_fast=False
                    )
                except Exception as e2:
                    print(f"加载processor失败，尝试降级方法: {str(e2)}")
                    try:
                        # 尝试直接加载基础的T5Tokenizer，绕过pre_tokenizer问题
                        from transformers import T5Tokenizer, Pix2StructImageProcessor
                        
                        # 手动创建tokenizer，避免使用有问题的pre_tokenizer配置
                        tokenizer = T5Tokenizer(
                            vocab_file=os.path.join(model_dir, "vocab.txt"),
                            merges_file=os.path.join(model_dir, "merges.txt"),
                            use_fast=False,
                            legacy=False
                        )
                        
                        # 加载image processor
                        image_processor = Pix2StructImageProcessor.from_pretrained(
                            model_dir,
                            local_files_only=True
                        )
                        
                        # 创建processor
                        from transformers import Pix2StructProcessor
                        self.processor = Pix2StructProcessor(
                            image_processor=image_processor,
                            tokenizer=tokenizer
                        )
                    except Exception as e3:
                        print(f"手动创建processor失败: {str(e3)}")
                        # 最后尝试最基本的加载方式，使用use_fast=False
                        self.processor = Pix2StructProcessor.from_pretrained(
                            model_dir,
                            local_files_only=True,
                            use_fast=False
                        )
            
            # 加载模型
            self.model = Pix2StructForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,
                local_files_only=True
            )
            
            # 将模型移到指定设备
            self.model.to(self.device)
            
            self.initialized = True
            print(f"Pix2Struct专家模块初始化成功，使用模型路径: {model_dir}")
            return self.model
            
        except ImportError as e:
            print(f"警告: 未安装必要的依赖，Pix2Struct专家模块不可用: {e}")
            print("安装命令: pip install transformers torch")
            return None
        except Exception as e:
            print(f"Pix2Struct专家模块初始化失败: {str(e)}")
            return None
    
    def process(self, image, question: Optional[str] = None) -> Dict[str, Any]:
        """处理图像生成HTML代码"""
        if not self.is_available():
            return {"html": "", "error": "Pix2Struct专家模块未初始化"}
        
        try:
            # 确保输入是PIL Image或转换为PIL Image
            if isinstance(image, torch.Tensor):
                # 处理张量：[C, H, W] -> [H, W, C]，并转换为0-255的uint8
                image = image.permute(1, 2, 0).cpu().numpy()
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            elif isinstance(image, np.ndarray):
                # 处理numpy数组
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                    image = image.transpose(1, 2, 0)
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"不支持的图像类型: {type(image)}")
            
            # 设置prompt
            prompt = "turn the pic into HTML"
            
            # 预处理图像
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # 生成HTML
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # 减少生成长度，提高速度
                    num_beams=2,  # 减少beam数量，提高速度
                    do_sample=False,  # 不使用采样，使用贪婪解码
                    early_stopping=True,  # 启用早停
                    length_penalty=1.0  # 长度惩罚
                )
            
            # 解码生成的文本
            generated_html = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            return {
                "html": generated_html,
                "prompt_used": prompt
            }
            
        except Exception as e:
            print(f"Pix2Struct处理失败: {str(e)}")
            return {"html": "", "error": f"Pix2Struct处理失败: {str(e)}"}
    
    def to_prompt(self, result: Dict[str, Any]) -> str:
        """转换为LLM提示词"""
        if "error" in result:
            return "HTML Structure: Unable to generate HTML"
        
        html = result.get("html", "")
        if not html:
            return "HTML Structure: No HTML generated"
        
        # 添加长度限制
        max_html_length = 1500
        if len(html) > max_html_length:
            html = html[:max_html_length] + "..."
        
        return f"HTML Structure: {html}"