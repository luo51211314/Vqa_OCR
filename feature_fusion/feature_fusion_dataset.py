import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import sys
import json
from PIL import Image
import io
import tempfile
import numpy as np
from typing import Dict, List, Any

# 添加项目路径
sys.path.append('/root/autodl-tmp/codes/Vqa_ocr')

# 导入真实特征提取器
try:
    from real_feature_extractor import RealFeatureExtractor
except ImportError:
    print("警告: 无法导入RealFeatureExtractor，将使用占位符特征提取器")
    RealFeatureExtractor = None

class ContrastiveFusionDataset(Dataset):
    """
    对比学习特征融合数据集
    从训练数据中提取LLaVA和专家模型的特征
    """
    
    def __init__(self, 
                 dataset_path: str,
                 expert_model: str = "paddleocr",
                 llava_model_path: str = "/root/autodl-tmp/model/llava_hug"):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.expert_model = expert_model
        self.llava_model_path = llava_model_path
        
        # 加载数据集
        self.samples = self._load_dataset()
        
        # 初始化特征提取器
        self.llava_feature_extractor = self._init_llava_feature_extractor()
        self.expert_feature_extractor = self._init_expert_feature_extractor()
    
    def _load_dataset(self) -> List[Dict]:
        """加载训练数据集"""
        samples = []
        
        # 检查Parquet文件
        train_parquet = os.path.join(self.dataset_path, "train_contrastive_fusion_train.parquet")
        val_parquet = os.path.join(self.dataset_path, "val_contrastive_fusion_validation.parquet")
        
        # 优先使用训练集
        if os.path.exists(train_parquet):
            try:
                import pandas as pd
                df = pd.read_parquet(train_parquet)
                
                for idx, row in df.iterrows():
                    # 保存图像字节数据，用于后续特征提取
                    image_bytes = row.get("image_bytes", None)
                    
                    # 创建临时图像文件路径
                    temp_image_path = f"/tmp/temp_image_{idx}.png"
                    
                    samples.append({
                        "image_bytes": image_bytes,
                        "temp_image_path": temp_image_path,
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "source": row.get("source", ""),
                        "ocr_text": row.get("ocr_text", ""),
                        "question_id": row.get("questionId", str(idx)),
                        "image_id": str(idx)
                    })
                print(f"从Parquet文件加载了 {len(samples)} 个训练样本")
                return samples
            except Exception as e:
                print(f"加载Parquet文件失败: {e}")
        
        # 检查验证集
        if os.path.exists(val_parquet):
            try:
                import pandas as pd
                df = pd.read_parquet(val_parquet)
                
                for idx, row in df.iterrows():
                    # 保存图像字节数据，用于后续特征提取
                    image_bytes = row.get("image_bytes", None)
                    
                    # 创建临时图像文件路径
                    temp_image_path = f"/tmp/temp_image_{idx}.png"
                    
                    samples.append({
                        "image_bytes": image_bytes,
                        "temp_image_path": temp_image_path,
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "source": row.get("source", ""),
                        "ocr_text": row.get("ocr_text", ""),
                        "question_id": row.get("questionId", str(idx)),
                        "image_id": str(idx)
                    })
                print(f"从验证集Parquet文件加载了 {len(samples)} 个训练样本")
                return samples
            except Exception as e:
                print(f"加载验证集Parquet文件失败: {e}")
        
        # 如果Parquet文件不存在，尝试JSON格式
        image_dir = os.path.join(self.dataset_path, "images")
        annotation_file = os.path.join(self.dataset_path, "annotations.json")
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                image_path = os.path.join(image_dir, ann["image_name"])
                if os.path.exists(image_path):
                    samples.append({
                        "image_path": image_path,
                        "question": ann.get("question", ""),
                        "answer": ann.get("answer", ""),
                        "image_id": ann.get("image_id", "")
                    })
        
        print(f"加载了 {len(samples)} 个训练样本")
        return samples
    
    def _init_llava_feature_extractor(self):
        """初始化LLaVA特征提取器"""
        try:
            if RealFeatureExtractor is not None:
                # 使用真实特征提取器
                extractor = RealFeatureExtractor(model_type="llava", model_path=self.llava_model_path)
                return extractor.extract_features
            else:
                # 使用占位符特征提取器
                def extract_llava_features(image_path, question):
                    return torch.randn(1, 256, 4096)  # [batch, seq_len, hidden_size]
                return extract_llava_features
        except Exception as e:
            print(f"LLaVA特征提取器初始化失败: {e}")
            return None
    
    def _init_expert_feature_extractor(self):
        """初始化专家模型特征提取器"""
        try:
            if self.expert_model == "paddleocr":
                return self._init_paddleocr_extractor()
            elif self.expert_model == "pix2struct":
                return self._init_pix2struct_extractor()
            else:
                raise ValueError(f"不支持的专家模型: {self.expert_model}")
        except Exception as e:
            print(f"专家模型特征提取器初始化失败: {e}")
            return None
    
    def _init_paddleocr_extractor(self):
        """初始化PaddleOCR特征提取器"""
        try:
            if RealFeatureExtractor is not None:
                # 使用真实特征提取器
                extractor = RealFeatureExtractor(model_type="paddleocr")
                return extractor.extract_features
            else:
                # 使用占位符特征提取器
                def extract_paddleocr_features(image_path):
                    return torch.randn(1, 128, 768)  # [batch, seq_len, hidden_size]
                return extract_paddleocr_features
        except Exception as e:
            print(f"PaddleOCR特征提取器初始化失败: {e}")
            return None
    
    def _init_pix2struct_extractor(self):
        """初始化Pix2Struct特征提取器"""
        try:
            if RealFeatureExtractor is not None:
                # 使用真实特征提取器
                extractor = RealFeatureExtractor(model_type="pix2struct")
                return extractor.extract_features
            else:
                # 使用占位符特征提取器
                def extract_pix2struct_features(image_path):
                    return torch.randn(1, 196, 768)  # [batch, seq_len, hidden_size]
                return extract_pix2struct_features
        except Exception as e:
            print(f"Pix2Struct特征提取器初始化失败: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # 处理图像数据
            if sample.get("image_bytes") is not None:
                # 从字节数据创建图像
                import io
                from PIL import Image
                
                image_bytes = sample["image_bytes"]
                if isinstance(image_bytes, bytes):
                    image = Image.open(io.BytesIO(image_bytes))
                    # 保存为临时文件用于特征提取
                    image.save(sample["temp_image_path"])
                    image_path = sample["temp_image_path"]
                else:
                    # 如果字节数据无效，使用随机特征
                    llava_features = torch.randn(256, 4096)
                    expert_features = torch.randn(128, 768)
            else:
                # 使用图像文件路径
                image_path = sample.get("image_path", "")
            
            # 提取特征
            if 'llava_features' not in locals():
                llava_features = self.llava_feature_extractor(
                    image_path, sample["question"]
                )
                expert_features = self.expert_feature_extractor(image_path)
            
            # 确保特征维度正确
            if len(llava_features.shape) == 1:
                llava_features = llava_features.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            elif len(llava_features.shape) == 2:
                llava_features = llava_features.unsqueeze(0)  # [1, seq_len, hidden_size]
            
            if len(expert_features.shape) == 1:
                expert_features = expert_features.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            elif len(expert_features.shape) == 2:
                expert_features = expert_features.unsqueeze(0)  # [1, seq_len, hidden_size]
            
            # 创建标签（用于对比学习）
            label = torch.tensor(idx, dtype=torch.long)
            
            return {
                "llava_features": llava_features.squeeze(0),  # [seq_len, hidden_size]
                "expert_features": expert_features.squeeze(0),  # [seq_len, expert_hidden_size]
                "labels": label,
                "image_id": sample["image_id"]
            }
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            # 返回一个默认样本
            return {
                "llava_features": torch.randn(256, 4096),
                "expert_features": torch.randn(128, 768),
                "labels": torch.tensor(idx, dtype=torch.long),
                "image_id": sample["image_id"]
            }