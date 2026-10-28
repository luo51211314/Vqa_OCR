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

                # 注意：引擎未返回置信度时保持为空，由消费方记为 None；
                # 不再伪造 0.9（伪造值会让置信度过滤产生静默偏差）。
                min_len = min(len(rec_texts), len(dt_polys))
                
                img_h, img_w = current_image_np.shape[0], current_image_np.shape[1]

                if rec_texts:
                    for j in range(min_len):
                        text = rec_texts[j]
                        score = float(rec_scores[j]) if j < len(rec_scores) else None
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

                            # 中心点归一化到 [0,1]（论文 p_i）
                            center_x = (minx + maxx) / 2 / img_w
                            center_y = (miny + maxy) / 2 / img_h
                            # 归一化 bbox（论文 B_i 的轴对齐表示）
                            nminx, nmaxx = minx / img_w, maxx / img_w
                            nminy, nmaxy = miny / img_h, maxy / img_h
                        except Exception:
                            minx, miny, maxx, maxy = 0.0, 0.0, 100.0, 30.0
                            center_x = center_y = 0.5
                            nminx, nminy, nmaxx, nmaxy = 0.0, 0.0, 1.0, 1.0

                        ocr_texts.append({
                            "text": text,
                            "confidence": score,
                            "center_x": float(center_x),
                            "center_y": float(center_y),
                            "minx": minx,
                            "maxx": maxx,
                            "miny": miny,
                            "maxy": maxy,
                            # 归一化几何量 + 图像尺寸：排序/坐标渲染/过滤统一使用
                            "nminx": float(nminx),
                            "nminy": float(nminy),
                            "nmaxx": float(nmaxx),
                            "nmaxy": float(nmaxy),
                            "image_width": int(img_w),
                            "image_height": int(img_h),
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
    
    def to_prompt(
        self,
        result: Dict[str, Any],
        mode: str = "native",
        question: str = "",
        template: str = "p2",
        budget_tokens: Optional[int] = None,
        token_counter=None,
        return_stats: bool = False,
    ):
        """OCR 结果 -> 模型输入 prompt。

        历史上本方法内置了 normal/random/LR-TB/direct 四套互相不一致的渲染
        逻辑和 3000 字符硬截断；现已全部收敛到唯一的序列化器
        ``expert.ocr_serializer.build_ocr_prompt``（论文 φ 的权威实现）。

        Args:
            result: ``process()`` 的 OCR 识别结果。
            mode: 顺序：native/random/ros/coordinate（兼容旧名 normal/LR-TB/direct）。
            question: 问题文本（模板内统一包装，不再由调用方二次拼接）。
            template: p1/p2/p3 文本模板。
            budget_tokens: OCR 段 token 预算，None/0 表示不限；超预算按整块丢弃。
            token_counter: 目标模型 tokenizer 计数函数。
            return_stats: True 时返回 (prompt, stats)，否则仅返回 prompt（兼容旧调用）。
        """
        from .ocr_serializer import build_ocr_prompt

        if "error" in result:
            prompt = "the ocr text: not found ocr text"
            return (prompt, {}) if return_stats else prompt

        ocr_texts = result.get("ocr_texts", [])
        out = build_ocr_prompt(
            ocr_texts,
            question=question,
            order=mode,
            template=template,
            budget_tokens=budget_tokens,
            token_counter=token_counter,
        )
        if return_stats:
            return out["prompt"], out["stats"]
        return out["prompt"]
