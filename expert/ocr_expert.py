import torch
from typing import Dict, Any, Optional
from .base_expert import BaseExpert
import os
import cv2
import numpy as np
from PIL import Image  # 导入PIL库来处理Image对象

class OcrExpert(BaseExpert):
    """OCR文本识别专家模块"""
    
    def __init__(self):
        super().__init__("ocr")
        self.model = None
        self.processor = None
    
    def initialize(self, model_path: Optional[str] = None, **kwargs):
        """初始化OCR模型"""
        try:
            from paddleocr import PaddleOCR
            
            # 使用PP-OCR模型
            model_dir = model_path or "/root/autodl-tmp/model/ppocr_hug"
            
            # 根据用户提供的参数列表初始化PaddleOCR
            self.model = PaddleOCR(
                text_detection_model_dir=os.path.join(model_dir, "det") if os.path.exists(os.path.join(model_dir, "det")) else None,
                text_recognition_model_dir=os.path.join(model_dir, "rec") if os.path.exists(os.path.join(model_dir, "rec")) else None,
                textline_orientation_model_dir=None,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="ch",
                ocr_version="PP-OCRv5"
            )
            
            self.initialized = True
            print(f"OCR专家模块初始化成功，使用模型路径: {model_dir}")
            
        except ImportError:
            print("警告: 未安装paddleocr，OCR专家模块不可用")
            print("安装命令: pip install paddleocr")
        except Exception as e:
            print(f"OCR专家模块初始化失败: {str(e)}")
            print("请检查本地模型文件是否完整")
    
    def process(self, image, question: Optional[str] = None) -> Dict[str, Any]:
        """处理图像进行OCR识别"""
        if not self.is_available():
            return {"ocr_text": "", "error": "OCR专家模块未初始化"}
        
        try:
            # 转换图像格式
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            elif isinstance(image, Image.Image):  # 处理PIL Image对象
                image = np.array(image)
            
            # 检查图像通道顺序和类型
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # 进行OCR识别
            result = self.model.ocr(image)
            
            # 提取识别文本
            ocr_texts = []
            if result and isinstance(result, list) and len(result) > 0:
                result_dict = result[0]
                
                # 提取识别文本和置信度
                if 'rec_texts' in result_dict and 'rec_scores' in result_dict:
                    rec_texts = result_dict['rec_texts']
                    rec_scores = result_dict['rec_scores']
                    
                    for text, score in zip(rec_texts, rec_scores):
                        ocr_texts.append({
                            "text": text,
                            "confidence": float(score)
                        })
            
            return {
                "ocr_texts": ocr_texts,
                "full_text": " ".join([item["text"] for item in ocr_texts]),
                "total_lines": len(ocr_texts)
            }
            
        except Exception as e:
            print(f"OCR处理失败: {str(e)}")
            return {"ocr_texts": [], "error": f"OCR处理失败: {str(e)}"}
    
    def to_prompt(self, result: Dict[str, Any]) -> str:
        """转换为LLM提示词"""
        if "error" in result:
            return "OCR Text: Unable to recognize text"
        
        ocr_texts = result.get("ocr_texts", [])
        if not ocr_texts:
            return "OCR Text: No text detected"
        
        # 构建英文结构化输出，包含行号但不换行
        extracted_texts = []
        for i, item in enumerate(ocr_texts, 1):
            text = item.get("text", "")
            extracted_texts.append(f"Line {i}: {text}")
        
        full_text = " ".join(extracted_texts)
        
        # 添加长度限制，避免序列过长
        max_text_length = 1500  # 限制OCR文本长度
        if len(full_text) > max_text_length:
            # print(f"警告: OCR文本长度 {len(full_text)} 超过限制 {max_text_length}，进行截断")
            full_text = full_text[:max_text_length] + "..."
        
        return f"OCR Text: {full_text}"
