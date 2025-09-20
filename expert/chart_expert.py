import torch
from typing import Dict, Any, Optional
from .base_expert import BaseExpert
import os

class ChartExpert(BaseExpert):
    """图表解析专家模块"""
    
    def __init__(self):
        super().__init__("chart")
        self.model = None
    
    def initialize(self, model_path: Optional[str] = None, **kwargs):
        """初始化图表解析模型"""
        # 在导入任何transformers模块之前强制设置环境变量
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        try:
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
            
            # 使用Pix2Struct模型
            model_name = model_path or "/root/autodl-tmp/model/pix2struct_hug"
            
            self.model = Pix2StructForConditionalGeneration.from_pretrained(
                model_name, 
                local_files_only=True
            )
            self.processor = Pix2StructProcessor.from_pretrained(
                model_name, 
                local_files_only=True
            )
            
            self.model.to(self.device)
            self.initialized = True
            print(f"图表专家模块初始化成功，使用模型: {model_name}")
            
        except ImportError:
            print("警告: 未安装transformers，图表专家模块不可用")
            print("安装命令: pip install transformers")
        except Exception as e:
            print(f"图表专家模块初始化失败: {str(e)}")
            print("请检查本地模型文件是否完整")
            print(f"详细错误信息: {repr(e)}")
    
    def process(self, image, question: Optional[str] = None) -> Dict[str, Any]:
        """解析图表图像"""
        if not self.is_available():
            return {"chart_data": {}, "error": "图表专家模块未初始化"}
        
        # 为Pix2Struct VQA模型添加header文本 - 修改为只提取结构不回答问题
        header_text = "Extract the chart structure and data:"
        
        # 处理图像，添加header参数 - 使用hf.mirror下载字体
        try:
            # 设置HF_ENDPOINT使用镜像
            original_hf_endpoint = os.environ.get("HF_ENDPOINT")
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            # 使用hf.mirror下载字体
            from huggingface_hub import hf_hub_download
            font_path = hf_hub_download("ybelkada/fonts", "Arial.TTF")
            
            inputs = self.processor(
                images=image, 
                text=header_text,
                return_tensors="pt",
                font_path=font_path
            ).to(self.device)
            
            # 恢复原始HF_ENDPOINT
            if original_hf_endpoint:
                os.environ["HF_ENDPOINT"] = original_hf_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)
                
        except Exception as e:
            print(f"警告: 图表处理失败: {str(e)}")
            # 如果使用字体路径失败，尝试不使用字体参数
            try:
                inputs = self.processor(
                    images=image, 
                    text=header_text,
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e2:
                print(f"警告: 备用处理也失败: {str(e2)}")
                return {"chart_data": {}, "error": f"图表解析失败: {str(e2)}"}
        
        # 生成描述
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4
        )
        
        # 解码结果
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "chart_type": "auto_detected",
            "structured_data": generated_text,
            "raw_output": generated_text
        }
    
    def to_prompt(self, result: Dict[str, Any]) -> str:
        """转换为LLM提示词"""
        if "error" in result:
            return "图表信息: 无法解析图表"
        
        structured_data = result.get("structured_data", "")
        return f"图表信息: {structured_data}"