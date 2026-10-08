from loaders import VqaDataset
import editdistance
import numpy as np
import pandas as pd
import os
from PIL import Image
import io
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Dataset(VqaDataset):
    name = "tablevqa"

    def __init__(self, split="test", **_):
        super().__init__(split)
        # 1. 本地 parquet 文件夹路径
        self.data_dir = "/root/autodl-tmp/dataset/tableVQA/data"
        # 2. 根据split选择对应的parquet文件
        if split == "train":
            parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("train") and f.endswith(".parquet")]
        elif split in ["validation", "val"]:
            parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("validation") and f.endswith(".parquet")]
            if not parquet_files:
                # 尝试寻找val前缀的文件
                parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("val") and f.endswith(".parquet")]
        elif split == "test":
            parquet_files = [f for f in os.listdir(self.data_dir) if f.startswith("test") and f.endswith(".parquet")]
        else:
            raise ValueError(f"不支持的split: {split}")
            
        if not parquet_files:
            raise FileNotFoundError(f"在 {self.data_dir} 未找到 {split} 的 parquet 文件")
            
        # 3. 用 pandas 顺序读取所有分片
        dfs = [pd.read_parquet(os.path.join(self.data_dir, f)) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        
        # 4. 展平qa列中的数据
        flattened_data = []
        for index, row in df.iterrows():
            qa_en = row['qa'].get('en', [])
            for i, qa in enumerate(qa_en):
                flattened_data.append({
                    'id': f"{row['id']}_{i}",
                    'image': row['image'],
                    'question': qa['question'],
                    'answer': qa['answer']
                })
        
        self.df = pd.DataFrame(flattened_data)
        
        # 5. 检查数据列名
        print(f"TableVQA数据集展平后列名: {list(self.df.columns)}")
        print(f"数据集大小: {len(self.df)} 条记录")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        
        # 加载图像
        img = self._load_image(sample["image"])
        
        # 获取问题和答案
        prompt = sample["question"]  # 只返回原始问题，不添加后缀
        
        # 处理答案
        answers = [sample["answer"]]
        
        # 获取问题ID
        question_id = sample.get("id", idx)
        
        return img, prompt, answers, {"questionId": question_id}
    
    def _load_image(self, image_field):
        """兼容 bytes / path / PIL.Image"""
        if isinstance(image_field, dict) and "bytes" in image_field:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if isinstance(image_field, str):
            return Image.open(image_field).convert("RGB")
        if isinstance(image_field, Image.Image):
            return image_field.convert("RGB")
        raise ValueError(f"无法识别的 image 字段类型: {type(image_field)}")
    
    @staticmethod
    def _metrics_relaxed_accuracy(preds, refs):
        """宽松准确率：包含连续字符串即正确"""
        relaxed_scores = []
        for pred, ref_list in zip(preds, refs):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            max_score = 0.0
            for ref in ref_list:
                ref = str(ref).strip()
                if not ref:
                    continue
                
                pred_str = str(pred).strip().lower()
                ref_str = ref.lower()
                
                # 检查预测是否包含参考答案的连续字符串
                if ref_str in pred_str:
                    max_score = 1.0
                    break
            
            relaxed_scores.append(max_score)
        
        return {
            "relaxed_accuracy": float(np.mean(relaxed_scores)),
            "relaxed_accuracy_scores": relaxed_scores,
            "total_samples": len(preds)
        }
    
    @staticmethod
    def _metrics_anls(preds, refs):
        """ANLS指标"""
        anls_scores = []
        for pred, ref_list in zip(preds, refs):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            max_anls = 0.0
            for ref in ref_list:
                ref = str(ref).strip()
                if not ref:
                    continue
                
                pred_str = str(pred).strip()
                edit_dist = editdistance.eval(pred_str.lower(), ref.lower())
                max_len = max(len(pred_str), len(ref))
                norm_dist = edit_dist / max_len if max_len > 0 else 0
                anls = max(0, 1 - norm_dist)
                max_anls = max(max_anls, anls)
            
            anls_scores.append(max_anls if max_anls >= 0.5 else 0.0)
        
        return {
            "anls": float(np.mean(anls_scores)),
            "anls_scores": anls_scores,
            "total_samples": len(preds)
        }
    
    @staticmethod
    def _metrics_bleu(preds, refs):
        """BLEU指标"""
        bleu_scores = []
        smooth_fn = SmoothingFunction().method1
        
        for pred, ref_list in zip(preds, refs):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            # 将参考答案转换为nltk需要的格式：list of token lists
            references = [str(ref).lower().split() for ref in ref_list]
            hypothesis = str(pred).lower().split()
            
            # 计算BLEU分数，使用平滑函数避免0分
            bleu_score = sentence_bleu(references, hypothesis, smoothing_function=smooth_fn)
            bleu_scores.append(bleu_score)
        
        return {
            "bleu": float(np.mean(bleu_scores)),
            "bleu_scores": bleu_scores,
            "total_samples": len(preds)
        }
    
    @staticmethod
    def metrics(preds, refs, metric_type="anls"):
        """
        支持多种评估指标
        metric_type: "anls", "relaxed_accuracy", "relaxed_accuracy_80", "bleu"
        """
        if metric_type == "anls":
            return Dataset._metrics_anls(preds, refs)
        elif metric_type == "relaxed_accuracy":
            return Dataset._metrics_relaxed_accuracy(preds, refs)
        elif metric_type == "relaxed_accuracy_80":
            return Dataset._metrics_relaxed_accuracy_80(preds, refs)
        elif metric_type == "bleu":
            return Dataset._metrics_bleu(preds, refs)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
    
    @staticmethod
    def _metrics_relaxed_accuracy_80(preds, refs):
        """80%字符匹配的宽松准确率"""
        relaxed_scores = []
        for pred, ref_list in zip(preds, refs):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            max_score = 0.0
            for ref in ref_list:
                ref = str(ref).strip()
                if not ref:
                    continue
                
                pred_str = str(pred).strip().lower()
                ref_str = ref.lower()
                
                # 计算字符重叠率
                pred_chars = set(pred_str)
                ref_chars = set(ref_str)
                
                if ref_chars:
                    overlap_ratio = len(pred_chars.intersection(ref_chars)) / len(ref_chars)
                    if overlap_ratio >= 0.8:  # 80%字符匹配
                        max_score = 1.0
                        break
            
            relaxed_scores.append(max_score)
        
        return {
            "relaxed_accuracy_80": float(np.mean(relaxed_scores)),
            "relaxed_accuracy_80_scores": relaxed_scores,
            "total_samples": len(preds)
        }
