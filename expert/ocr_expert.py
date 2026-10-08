import torch
from typing import Dict, Any, Optional, List
from .base_expert import BaseExpert
import os
import numpy as np
from PIL import Image

class OcrExpert(BaseExpert):
    """OCR文本识别专家模块"""
    
    def __init__(self):
        super().__init__("ocr")
        self.model = None
        self.processor = None
        self.preloaded_models: Dict[str, Any] = {}
        self.ocr_pipeline = None
    
    def initialize(self, model_path: Optional[str] = None, **kwargs):
        """初始化OCR模型"""
        try:
            from paddleocr import PaddleOCR
            
            model_dir = model_path or "/root/autodl-tmp/model/ppocr_hug"
            
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
            return self.model
            
        except ImportError:
            print("警告: 未安装paddleocr，OCR专家模块不可用")
            print("安装命令: pip install paddleocr")
            return None
        except Exception as e:
            print(f"OCR专家模块初始化失败: {str(e)}")
            print("请检查本地模型文件是否完整")
    
    def set_preloaded_models(self, models: Dict[str, Any]):
        """设置预加载的BERT模型
        
        Args:
            models: 预加载的模型字典
        """
        self.preloaded_models = models
        print(f"[OcrExpert] 接收到 {len(models)} 个预加载模型")
    
    def initialize_pipeline(self, mode: str = "full", debug: bool = True):
        """初始化OCR处理流水线
        
        Args:
            mode: 流水线模式
            debug: 是否输出调试信息
        """
        try:
            import sys
            algorithm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Algorithm")
            if algorithm_path not in sys.path:
                sys.path.insert(0, algorithm_path)
            
            from ocr_pipeline import create_pipeline
            
            self.ocr_pipeline = create_pipeline(
                mode=mode, 
                debug=debug,
                preloaded_models=self.preloaded_models
            )
        except Exception as e:
            print(f"[OcrExpert] OCR流水线初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.ocr_pipeline = None
            return None
    
    def process(self, image, question: Optional[str] = None) -> Dict[str, Any]:
        """处理图像进行OCR识别"""
        if not self.is_available():
            return {"ocr_texts": [], "error": "OCR专家模块未初始化"}
        
        try:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            current_image_np = image
            
            result = self.model.ocr(image)
            
            ocr_texts = []
            if result and isinstance(result, list) and len(result) > 0:
                ocr_result = result[0]
                
                rec_texts = []
                rec_scores = []
                dt_polys = []
                
                if hasattr(ocr_result, 'get'):
                    rec_texts = ocr_result.get('rec_texts', [])
                    rec_scores = ocr_result.get('rec_scores', [])
                    dt_polys = ocr_result.get('dt_polys', [])
                
                if not rec_texts and hasattr(ocr_result, 'rec_texts'):
                    rec_texts = getattr(ocr_result, 'rec_texts', [])
                    rec_scores = getattr(ocr_result, 'rec_scores', [])
                    dt_polys = getattr(ocr_result, 'dt_polys', [])
                
                if not rec_texts and isinstance(ocr_result, list):
                    for item in ocr_result:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            polygon = item[0]
                            if len(item) > 1 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                                text = item[1][0]
                                score = item[1][1]
                                rec_texts.append(text)
                                rec_scores.append(score)
                                dt_polys.append(polygon)
                
                if not dt_polys and rec_texts:
                    dt_polys = [[[j*100, 0, j*100+100, 0, j*100+100, 30, j*100, 30]] for j in range(len(rec_texts))]
                
                if not rec_scores and rec_texts:
                    rec_scores = [0.9 for _ in range(len(rec_texts))]
                
                min_len = min(len(rec_texts), len(rec_scores), len(dt_polys))
                
                if rec_texts:
                    for j in range(min_len):
                        text = rec_texts[j]
                        score = rec_scores[j] if j < len(rec_scores) else 0.0
                        polygon = dt_polys[j] if j < len(dt_polys) else [[0, 0, 100, 0, 100, 30, 0, 30]]
                        
                        try:
                            polygon_np = np.array(polygon)
                            if len(polygon_np.shape) == 3:
                                polygon_np = polygon_np[0]
                            
                            x_coords = polygon_np[:, 0]
                            y_coords = polygon_np[:, 1]
                            minx = float(np.min(x_coords))
                            maxx = float(np.max(x_coords))
                            miny = float(np.min(y_coords))
                            maxy = float(np.max(y_coords))
                            
                            center_x = (minx + maxx) / 2 / current_image_np.shape[1]
                            center_y = (miny + maxy) / 2 / current_image_np.shape[0]
                        except Exception as e:
                            minx, miny, maxx, maxy = 0.0, 0.0, 100.0, 30.0
                            center_x = 0.5
                            center_y = 0.5
                        
                        ocr_texts.append({
                            "text": text,
                            "confidence": float(score),
                            "center_x": float(center_x),
                            "center_y": float(center_y),
                            "minx": minx,
                            "maxx": maxx,
                            "miny": miny,
                            "maxy": maxy,
                            "polygon": polygon
                        })
            
            return {
                "ocr_texts": ocr_texts,
                "full_text": " ".join([item["text"] for item in ocr_texts]),
                "total_lines": len(ocr_texts)
            }
            
        except Exception as e:
            print(f"OCR处理失败: {str(e)}")
            return {"ocr_texts": [], "error": f"OCR处理失败: {str(e)}"}
    
    def to_prompt(self, result: Dict[str, Any], mode: str = "normal") -> str:
        """转换为LLM提示词（不压缩）
        
        Args:
            result: OCR识别结果
            mode: OCR处理模式，可选值：
                - normal: 默认模式，带置信度筛选（不考虑置信度，全部保留）
                - random: 随机打乱 OCR block 顺序（seed=42），然后逗号拼接
                - LR-TB: 按列优先组合（按中心点x坐标分组，然后在每组内按y排序）
                - direct: 不用文字拼接，直接将 block 信息作为 prompt 输出
            
        Returns:
            prompt字符串
        """
        import random
        
        if "error" in result:
            return "the ocr text: not found ocr text"
        
        ocr_texts = result.get("ocr_texts", [])
        if not ocr_texts:
            return "the ocr text: not found ocr text"
        
        # 先全部保留，不考虑置信度
        all_texts = [item.copy() for item in ocr_texts]
        
        if mode == "normal":
            # 默认模式，不考虑置信度，全部保留，按原始顺序输出
            texts = [item.get("text", "") for item in all_texts]
            full_text = ", ".join(texts)
            prompt = f"the ocr text: {full_text}"
        elif mode == "random":
            # 随机打乱 OCR block 顺序（seed=42），然后逗号拼接
            random.seed(42)
            shuffled_texts = all_texts.copy()
            random.shuffle(shuffled_texts)
            texts = [item.get("text", "") for item in shuffled_texts]
            full_text = ", ".join(texts)
            prompt = f"the ocr text: {full_text}"
        elif mode == "LR-TB":
            # 按列优先组合（按中心点x坐标分组，然后在每组内按y排序）
            # 先按 x 坐标分桶（假设图片宽度归一化，分 10 个桶）
            sorted_by_x = sorted(all_texts, key=lambda x: x.get("center_x", 0.5))
            # 然后按 x 相近的分组，在组内按 y 排序
            # 简单起见，直接按 x 排序，然后按 y 排序
            lr_tb_texts = sorted(sorted_by_x, key=lambda x: (round(x.get("center_x", 0.5), 2), x.get("center_y", 0.5)))
            texts = [item.get("text", "") for item in lr_tb_texts]
            full_text = ", ".join(texts)
            prompt = f"the ocr text: {full_text}"
        elif mode == "direct":
            # 不用文字拼接，直接将 block 信息作为 prompt 输出，不包含 confidence
            blocks = []
            for idx, item in enumerate(all_texts):
                blocks.append(f"block {idx+1}: text={item.get('text', '')}, pos=({item.get('minx', 0.0):.4f}, {item.get('miny', 0.0):.4f}, {item.get('maxx', 1.0):.4f}, {item.get('maxy', 1.0):.4f})")
            full_text = "\n".join(blocks)
            prompt = f"the ocr blocks:\n{full_text}"
        else:
            # 默认模式
            texts = [item.get("text", "") for item in all_texts]
            full_text = ", ".join(texts)
            prompt = f"the ocr text: {full_text}"
        
        max_text_length = 3000
        if len(prompt) > max_text_length:
            prompt = prompt[:max_text_length] + "..."
        
        return prompt
