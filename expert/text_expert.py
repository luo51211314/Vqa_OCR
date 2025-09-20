import torch
import cv2
import numpy as np
from typing import Dict, Any, Optional
from .base_expert import BaseExpert

class TextExpert(BaseExpert):
    """文本检测与识别专家模块"""
    
    def __init__(self):
        super().__init__("text")
        self.detector = None  # DBNet++ 检测器
        self.recognizer = None  # SVTR 识别器
    
    def initialize(self, model_path: Optional[str] = None, **kwargs):
        """初始化文本检测和识别模型"""
        try:
            # 这里需要安装相应的库：pip install dbnetpp svtr
            from dbnetpp import DBNetPP
            from svtr import SVTR
            
            # 初始化检测器
            self.detector = DBNetPP(pretrained=True)
            
            # 初始化识别器  
            self.recognizer = SVTR(pretrained=True)
            
            self.initialized = True
            print(f"文本专家模块初始化成功")
            
        except ImportError:
            print("警告: 未安装dbnetpp或svtr，文本专家模块不可用")
            print("安装命令: pip install dbnetpp svtr")
    
    def process(self, image, question: Optional[str] = None) -> Dict[str, Any]:
        """处理图像中的文本"""
        if not self.is_available():
            return {"text_regions": [], "error": "文本专家模块未初始化"}
        
        # 转换为numpy数组
        if hasattr(image, 'numpy'):
            img_np = image.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(image)
        
        # 文本检测
        boxes = self.detector.detect(img_np)
        
        # 文本识别
        text_regions = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cropped = img_np[y1:y2, x1:x2]
            
            if cropped.size > 0:
                text = self.recognizer.recognize(cropped)
                text_regions.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "text": text,
                    "confidence": 0.9  # 示例值
                })
        
        return {
            "text_regions": text_regions,
            "total_regions": len(text_regions)
        }
    
    def to_prompt(self, result: Dict[str, Any]) -> str:
        """转换为LLM提示词"""
        if "error" in result:
            return "文本信息: 无法提取文本"
        
        text_regions = result.get("text_regions", [])
        if not text_regions:
            return "文本信息: 未检测到文本"
        
        prompt_lines = ["文本信息:"]
        for i, region in enumerate(text_regions, 1):
            bbox = region["bbox"]
            text = region["text"]
            prompt_lines.append(f"区域{i}(坐标[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]): '{text}'")
        
        return "\n".join(prompt_lines)