from loaders import VqaDataset
import numpy as np
import pandas as pd
import os
from PIL import Image
import io
import editdistance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Dataset(VqaDataset):
    name = "testvqa"

    def __init__(self, split="test", **_):
        super().__init__(split)
        # 本地 parquet 文件夹路径
        self.data_dir = "/root/autodl-tmp/dataset/testVQA"
        # 加载 test_dataset.parquet 文件
        parquet_file = os.path.join(self.data_dir, "test_dataset.parquet")
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"在 {self.data_dir} 未找到 test_dataset.parquet 文件")
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img = self._load_image(sample["image_bytes"])
        prompt = sample['question']  # 只返回原始问题
        answers = [sample["answer"]]  # 转换为 List[str] 格式
        return img, prompt, answers, {"question_type": sample["question_type"]}
    
    def _load_image(self, image_field):
        # 兼容 bytes / path / PIL.Image
        if isinstance(image_field, bytes):
            return Image.open(io.BytesIO(image_field)).convert("RGB")
        if isinstance(image_field, str):
            return Image.open(image_field).convert("RGB")
        if isinstance(image_field, Image.Image):
            return image_field.convert("RGB")
        raise ValueError(f"无法识别的 image 字段类型: {type(image_field)}")
    
    @staticmethod
    def _metrics_accuracy(preds, refs):
        """标准准确率：完全匹配或包含即正确"""
        scores = []
        for pred, ref_list in zip(preds, refs):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            pred_str = str(pred).strip().lower()
            is_correct = False
            
            for ref in ref_list:
                ref_str = str(ref).strip().lower()
                if not ref_str:
                    continue
                
                # 检查预测是否完全匹配或包含参考答案
                if pred_str == ref_str or ref_str in pred_str:
                    is_correct = True
                    break
            
            scores.append(1.0 if is_correct else 0.0)
        
        return {
            "accuracy": float(np.mean(scores)),
            "scores": scores,
            "total_samples": len(preds)
        }
    
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
        smoothie = SmoothingFunction().method4
        
        for pred, ref_list in zip(preds, refs):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            # 将参考文本和预测文本拆分为单词
            pred_tokens = str(pred).strip().lower().split()
            
            # 将每个参考转换为单词列表
            ref_tokens_list = []
            for ref in ref_list:
                ref_str = str(ref).strip().lower()
                if ref_str:
                    ref_tokens_list.append(ref_str.split())
            
            if not ref_tokens_list:
                # 如果没有有效的参考，跳过
                bleu_scores.append(0.0)
                continue
            
            # 计算BLEU分数，使用平滑函数避免零分
            try:
                bleu_score = sentence_bleu(ref_tokens_list, pred_tokens, smoothing_function=smoothie)
            except:
                bleu_score = 0.0
            
            bleu_scores.append(bleu_score)
        
        return {
            "bleu": float(np.mean(bleu_scores)),
            "bleu_scores": bleu_scores,
            "total_samples": len(preds)
        }
    
    @staticmethod
    def metrics(preds, refs, metric_type="accuracy"):
        """
        支持多种评估指标
        metric_type: "accuracy", "relaxed_accuracy", "anls", "bleu"
        """
        if metric_type == "accuracy":
            return Dataset._metrics_accuracy(preds, refs)
        elif metric_type == "relaxed_accuracy":
            return Dataset._metrics_relaxed_accuracy(preds, refs)
        elif metric_type == "anls":
            return Dataset._metrics_anls(preds, refs)
        elif metric_type == "bleu":
            return Dataset._metrics_bleu(preds, refs)
        else:
            raise ValueError(f"不支持的指标类型: {metric_type}")