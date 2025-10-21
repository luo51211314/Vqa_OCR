import torch
import numpy as np
from PIL import Image
import os
import torch
from tqdm import tqdm
import json
from llava.mm_utils import process_images

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.llava_model = None
        self.ocr_model = None
        self.initialized = False
        
    def initialize(self, llava_model=None, tokenizer=None):
        """初始化特征提取器，优先使用传入的LLaVA模型，否则加载新模型
        
        Args:
            llava_model: 已初始化的LLaVA模型实例，如果提供则使用此模型
            tokenizer: 已初始化的tokenizer实例，如果提供则使用此tokenizer
        """
        if not self.initialized:
            # 使用传入的模型和tokenizer，避免重复加载
            if llava_model is not None and tokenizer is not None:
                print("使用传入的LLaVA模型和tokenizer")
                self.llava_model = llava_model
                self.tokenizer = tokenizer
            else:
                # 原始加载逻辑，以保持向后兼容性
                print("加载新的LLaVA模型和tokenizer")
                from llava.model import LlavaLlamaForCausalLM
                from transformers import LlamaTokenizer
                from llava.mm_utils import get_model_name_from_path
                
                model_name = get_model_name_from_path(self.config['model_config']['llava_model_path'])
                tokenizer = LlamaTokenizer.from_pretrained(
                    self.config['model_config']['llava_model_path'],
                    cache_dir=self.config['training_config'].get('cache_dir', None),
                    model_max_length=self.config['training_config']['max_length'],
                    padding_side="right",
                    use_fast=False,
                )
                
                model = LlavaLlamaForCausalLM.from_pretrained(
                    self.config['model_config']['llava_model_path'],
                    cache_dir=self.config['training_config'].get('cache_dir', None),
                    dtype=torch.float32,
                    device_map="auto",
                )
                
                self.llava_model = model
                self.tokenizer = tokenizer
                
            # 初始化OCR模型
            try:
                from paddleocr import PaddleOCR
                ocr_model = PaddleOCR(
                    text_detection_model_dir=os.path.join(self.config['model_config']['ocr_model_path'], "det") \
                        if os.path.exists(os.path.join(self.config['model_config']['ocr_model_path'], "det")) else None,
                    text_recognition_model_dir=os.path.join(self.config['model_config']['ocr_model_path'], "rec") \
                        if os.path.exists(os.path.join(self.config['model_config']['ocr_model_path'], "rec")) else None,
                    textline_orientation_model_dir=None,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang="ch",
                    ocr_version="PP-OCRv5"
                )
            except Exception as e:
                print(f"OCR模型初始化失败: {str(e)}")
                ocr_model = None
            
            self.ocr_model = ocr_model
            self.initialized = True
            
        return self.llava_model, self.tokenizer, self.ocr_model
        
    def extract_llava_features(self, image):
        """从LLaVA模型中提取视觉特征
        
        Args:
            image: 输入图像，可以是PIL Image对象或PIL Image对象列表（批次）
            
        Returns:
            features: 提取的视觉特征
        """
        if not self.initialized:
            self.initialize()
        
        # 确保是PIL图像或PIL图像列表
        if isinstance(image, Image.Image):
            # 单张图像，包装为列表
            images_list = [image]
        elif isinstance(image, list) and all(isinstance(img, Image.Image) for img in image):
            # 批次图像
            images_list = image
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        # 使用process_images处理PIL图像
        from llava.mm_utils import process_images
        
        image_processor = self.llava_model.get_vision_tower().image_processor
        image_tensor = process_images(images_list, image_processor, self.llava_model.config)
        image_tensor = image_tensor.to(self.llava_model.device)
        
        # 处理形状为(1,1,H,W)的四维张量
        if len(image_tensor.shape) == 4:
            # 检查通道维度是否正确
            if image_tensor.shape[1] not in [1, 3]:
                # 如果通道维度不在正确的位置，尝试调整
                if image_tensor.shape[0] in [1, 3]:
                    # 假设格式是(channels, height, width)，添加批量维度
                    image_tensor = image_tensor.unsqueeze(0)
                elif image_tensor.shape[0] == 1 and image_tensor.shape[1] == 1:
                    # 特殊处理形状为(1,1,H,W)的张量
                    image_tensor = image_tensor.squeeze(1)  # 移除第二个维度
        
        # 确保图像具有正确的通道数和数据类型
        if image_tensor.shape[1] == 1:
            # 如果是单通道图像，复制到3通道
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
            
        # 确保视觉塔已加载
        vision_tower = self.llava_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        
        # 获取投影层
        projector = self.llava_model.get_model().mm_projector
        
        # 直接使用vision_tower进行特征提取，但避免直接调用它的forward方法
        # 而是手动处理图像并调用内部的vision_tower模型
        # 获取内部的vision_tower模型（CLIPVisionModel）
        internal_vision_model = vision_tower.vision_tower
        
        # 视觉特征提取部分使用no_grad以节省内存
        with torch.no_grad():
            # 处理图像并提取特征
            image_forward_outs = internal_vision_model(image_tensor.to(device=internal_vision_model.device, dtype=internal_vision_model.dtype), 
                                                       output_hidden_states=True)
            
            # 应用特征选择
            select_layer = vision_tower.select_layer
            select_feature = vision_tower.select_feature
            image_features = image_forward_outs.hidden_states[select_layer]
            
            if select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif select_feature == 'cls_patch':
                pass  # 不做任何操作
            else:
                raise ValueError(f'Unexpected select feature: {select_feature}')
            
            # 确保图像特征和投影器在同一个设备上，并且数据类型匹配
            model_device = next(self.llava_model.parameters()).device
            model_dtype = next(self.llava_model.parameters()).dtype
            image_features = image_features.to(model_device, dtype=model_dtype)
        
        # 投影层操作移出no_grad上下文，以允许梯度计算
        image_features = projector(image_features)
        
        # 如果特征是3D的 [batch_size, num_patches, hidden_size]，则求平均
        if len(image_features.shape) == 3:
            # 对patches维度求平均
            avg_features = image_features.mean(dim=1)
            return avg_features
        else:
            return image_features
        
    def extract_ocr_features(self, image):
        """提取OCR文本及其位置信息，支持批次处理（直接处理原始PIL图像）"""
        if not self.initialized:
            self.initialize()
        
        # 检查是否为批次图像（列表形式的PIL图像）
        from PIL import Image  # 确保在生成器表达式中可见
        is_batch = isinstance(image, list) and all(isinstance(img, Image.Image) for img in image)
        batch_size = len(image) if is_batch else 1
        batch_results = []
        
        # 遍历批次中的每个图像
        for i in range(batch_size):
            # 获取当前图像
            if is_batch:
                current_image = image[i]
            else:
                current_image = image
            
            # 确保是PIL图像
            if isinstance(current_image, Image.Image):
                current_image_np = np.array(current_image)
            elif isinstance(image, np.ndarray):
                current_image_np = current_image
            else:
                raise TypeError(f"不支持的图像类型: {type(current_image)}")

            # 确保数据类型正确
            if current_image_np.dtype != np.uint8:
                current_image_np = (current_image_np * 255).astype(np.uint8)
            
            # 进行OCR识别
            ocr_texts = []
            full_description = ""
            
            if self.ocr_model is not None:
                try:
                    # 使用predict方法（推荐）而不是ocr方法
                    # # 保存图片以检查是否已损坏
                    # try:
                    #     if isinstance(current_image_np, np.ndarray):
                    #         from PIL import Image
                    #         img = Image.fromarray(current_image_np)
                    #         img.save("temp_check_image.png")
                    #         print("图片已保存为 temp_check_image.png，可检查是否损坏")
                    # except Exception as e:
                    #     print(f"保存图片时出错，可能图片已损坏: {str(e)}")
                    result = self.ocr_model.predict(current_image_np)
                    
                    # 提取识别文本和位置信息
                    if result and isinstance(result, list) and len(result) > 0:
                        # 处理PaddleOCR返回的OCRResult对象
                        ocr_result = result[0]
                        
                        # 尝试多种方式访问识别结果
                        rec_texts = []
                        rec_scores = []
                        dt_polys = []
                        
                        # 方法1: 尝试使用字典风格的访问方式（get方法）
                        if hasattr(ocr_result, 'get'):
                            rec_texts = ocr_result.get('rec_texts', [])
                            rec_scores = ocr_result.get('rec_scores', [])
                            dt_polys = ocr_result.get('dt_polys', [])
                        
                        # 方法2: 如果字典风格访问失败，尝试直接属性访问
                        if not rec_texts and hasattr(ocr_result, 'rec_texts'):
                            rec_texts = getattr(ocr_result, 'rec_texts', [])
                            rec_scores = getattr(ocr_result, 'rec_scores', [])
                            dt_polys = getattr(ocr_result, 'dt_polys', [])
                        
                        # 方法3: 尝试从OCRResult对象中提取
                        if not rec_texts and isinstance(ocr_result, list):
                            # 检查是否是直接的结果列表
                            for item in ocr_result:
                                if isinstance(item, (list, tuple)) and len(item) >= 2:
                                    # 通常OCR结果格式: [[[[x1,y1],[x2,y2],...], (text, score)], ...]
                                    polygon = item[0]
                                    if len(item) > 1 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                                        text = item[1][0]
                                        score = item[1][1]
                                        rec_texts.append(text)
                                        rec_scores.append(score)
                                        dt_polys.append(polygon)
                        
                        # 如果没有检测框，但有文本，创建默认的检测框
                        if not dt_polys and rec_texts:
                            # 为每个文本创建一个默认的检测框
                            dt_polys = [[[j*100, 0, j*100+100, 0, j*100+100, 30, j*100, 30]] for j in range(len(rec_texts))]
                        
                        # 如果没有置信度，但有文本，创建默认的置信度
                        if not rec_scores and rec_texts:
                            rec_scores = [0.9 for _ in range(len(rec_texts))]  # 默认置信度0.9
                        
                        # 确保各列表长度匹配
                        min_len = min(len(rec_texts), len(rec_scores), len(dt_polys))
                        
                        # 处理识别到的文本
                        if rec_texts:
                            for j in range(min_len):
                                text = rec_texts[j]
                                score = rec_scores[j] if j < len(rec_scores) else 0.0
                                polygon = dt_polys[j] if j < len(dt_polys) else [[0, 0, 100, 0, 100, 30, 0, 30]]
                                
                                # 计算文本的中心点坐标
                                try:
                                    polygon_np = np.array(polygon)
                                    # 处理可能的嵌套结构
                                    if len(polygon_np.shape) == 3:
                                        polygon_np = polygon_np[0]  # 取第一个多边形
                                    
                                    center_x = np.mean(polygon_np[:, 0]) / current_image_np.shape[1]  # 归一化到[0, 1]
                                    center_y = np.mean(polygon_np[:, 1]) / current_image_np.shape[0]  # 归一化到[0, 1]
                                except Exception as e:
                                    # 如果计算中心点失败，使用默认值
                                    center_x = 0.5
                                    center_y = 0.5
                                    pass
                                
                                ocr_texts.append({
                                    "text": text,
                                    "confidence": float(score),
                                    "center_x": float(center_x),
                                    "center_y": float(center_y),
                                    "polygon": polygon
                                })
                except Exception as e:
                    # OCR处理失败，使用空结果
                    pass
            
            # 构造描述性文本
            full_description = (
                "the ocr text(format:'text, center pos(x,y)'): " +
                ", ".join(
                    f"'{item['text']}', ({item['center_x']:.2f}, {item['center_y']:.2f})"
                    for item in ocr_texts
                )
                if ocr_texts
                else "not found ocr text"
            )
            
            # # 将描述性文本写入txt文件
            # with open('ocr_description.txt', 'a', encoding='utf-8') as f:
            #     f.write(full_description + '\n')
            
            # 添加当前图像的OCR结果到批次结果
            batch_results.append({
                "ocr_texts": ocr_texts,
                "full_description": full_description
            })
        
        # 如果是单图像输入，返回单个结果；否则返回批次结果
        if is_batch:
            return batch_results
        else:
            return batch_results[0]
        
    def preprocess_image(self, image):
        """预处理图像以适应LLaVA模型"""
        from llava.train.dataset import process_images
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
        
        model_name = get_model_name_from_path(self.config['model_config']['llava_model_path'])
        image_tensor = process_images([image], self.tokenizer, self.config['model_config'], model_name)
        return image_tensor

    def text_to_embedding(self, text):
        """将文本转换为嵌入向量"""
        if not self.initialized:
            self.initialize()
        
        # 获取模型设备
        model_device = next(self.llava_model.parameters()).device
        
        # 使用tokenizer编码文本
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(model_device)
        
        # 获取嵌入向量 - 强制使用FP32
        # 获取模型的嵌入层
        embedding_layer = self.llava_model.get_model().embed_tokens
        # 确保嵌入层在正确的设备上运行
        if embedding_layer.weight.device != model_device:
            embedding_layer = embedding_layer.to(model_device)
        # 执行嵌入操作并强制转换为FP32
        text_embedding = embedding_layer(input_ids).to(torch.float32)
        # 对嵌入向量求平均，得到句子级别的表示
        text_embedding = text_embedding.mean(dim=1)
        
        return text_embedding

if __name__ == "__main__":
    # 测试特征提取器
    with open("config.json", "r") as f:
        config = json.load(f)
    
    extractor = FeatureExtractor(config)
    extractor.initialize()
    
    # 这里可以添加测试代码，比如加载一张图像并提取特征
    print("特征提取器初始化成功")