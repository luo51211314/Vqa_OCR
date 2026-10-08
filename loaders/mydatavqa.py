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
    name = "mydatavqa"

    def __init__(self, split="test", **_):
        super().__init__(split)
        # 1. 本地 parquet 文件夹路径
        self.data_dir = f"/root/autodl-tmp/dataset/MyDataVQA"
        # 2. 列出所有 split 对应分片
        if split == "test":
            parquet_files = sorted([f for f in os.listdir(self.data_dir)
                                    if f.startswith("test_dataset_batch") and f.endswith(".parquet")])
        elif split == "":
            # 空字符串表示测试所有文件
            parquet_files = sorted([f for f in os.listdir(self.data_dir)
                                    if f.endswith(".parquet")])
        else:
            parquet_files = sorted([f for f in os.listdir(self.data_dir)
                                    if f.startswith("batch_") and f.endswith(".parquet") and not f.startswith("test")])
        
        if not parquet_files:
            raise FileNotFoundError(f"在 {self.data_dir} 未找到 {split} 的 parquet 文件")
        # 3. 用 pandas 顺序读取所有分片
        dfs = [pd.read_parquet(os.path.join(self.data_dir, f)) for f in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img = self._load_image(sample["ImageBytes"])
        prompt = sample['question']  # 只返回原始问题，不添加后缀
        answers = [sample["answer_concise"]]  # List[str]
        return img, prompt, answers, {
            "answer_full": sample["answer_full"],
            "question_type": sample["question_type"],
            "origin": sample["origin"]
        }
    
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
        """BLEU指标 - 使用字符串分割代替NLTK分词器"""
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
    
    @staticmethod
    def _metrics_relaxed_acc_keywords(preds, refs):
        """
        基于关键词的宽松准确率：
        - 将 concise 以逗号为分隔分割成逗号个数+1个keywords
        - 若生成的答案包含第i个keywords（不区分大小写），+1分
        - 最后的分数除以keywords总数即为relaxed_acc_keywords分数
        """
        relaxed_scores = []
        for pred, ref_list in zip(preds, refs):
            if isinstance(ref_list, str):
                ref_list = [ref_list]
            
            max_score = 0.0
            for ref in ref_list:
                ref = str(ref).strip()
                if not ref:
                    continue
                
                # 将参考答案按逗号分割为关键词
                keywords = [k.strip() for k in ref.split(",") if k.strip()]
                if not keywords:
                    # 如果没有关键词，认为是0分
                    continue
                
                pred_str = str(pred).strip().lower()
                
                # 计算匹配的关键词数量
                matched_count = 0
                for keyword in keywords:
                    # 不考虑关键词末尾的句号
                    keyword = keyword.lower().rstrip('.')
                    # 不考虑预测答案末尾的句号
                    pred_str_no_dot = pred_str.rstrip('.')
                    if keyword in pred_str_no_dot:
                        matched_count += 1
                
                # 计算当前参考的得分
                score = matched_count / len(keywords)
                max_score = max(max_score, score)
            
            relaxed_scores.append(max_score)
        
        return {
            "relaxed_acc_keywords": float(np.mean(relaxed_scores)),
            "relaxed_acc_keywords_scores": relaxed_scores,
            "total_samples": len(preds)
        }
    
    @staticmethod
    def metrics(preds, refs, extra_info=None, lambda1=0.6, metric_type="weighted"):
        """
        支持多种评估指标
        metric_type: "anls", "relaxed_accuracy", "relaxed_accuracy_80", "bleu", "relaxed_acc_keywords", "weighted"
        weighted: λ₁*relaxed_acc_keywords(answer_concise) + λ₂*bleu(answer_full)，λ₁+λ₂=1
        """
        if metric_type == "anls":
            return Dataset._metrics_anls(preds, refs)
        elif metric_type == "relaxed_accuracy":
            return Dataset._metrics_relaxed_accuracy(preds, refs)
        elif metric_type == "relaxed_accuracy_80":
            return Dataset._metrics_relaxed_accuracy_80(preds, refs)
        elif metric_type == "bleu":
            return Dataset._metrics_bleu(preds, refs)
        elif metric_type == "relaxed_acc_keywords":
            return Dataset._metrics_relaxed_acc_keywords(preds, refs)
        elif metric_type == "weighted":
            # 获取 answer_full 用于 BLEU 计算
            if extra_info is None or len(extra_info) == 0 or "answer_full" not in extra_info[0]:
                raise ValueError("加权指标需要 answer_full 信息")
            
            answer_full_list = [info["answer_full"] for info in extra_info]
            
            # 计算 relaxed_acc_keywords (针对 answer_concise)
            relaxed_acc_keywords_result = Dataset._metrics_relaxed_acc_keywords(preds, refs)
            relaxed_acc_keywords_score = relaxed_acc_keywords_result["relaxed_acc_keywords"]
            relaxed_acc_keywords_scores = relaxed_acc_keywords_result["relaxed_acc_keywords_scores"]
            
            # 计算 BLEU (针对 answer_full) - 使用新的基于split的BLEU计算
            bleu_result = Dataset._metrics_bleu(preds, answer_full_list)
            bleu_score = bleu_result["bleu"]
            bleu_scores = bleu_result["bleu_scores"]
            
            # 计算每个样本的加权分数
            lambda2 = 1 - lambda1
            weighted_scores = [lambda1 * relaxed_acc_keyword + lambda2 * bleu for relaxed_acc_keyword, bleu in zip(relaxed_acc_keywords_scores, bleu_scores)]
            weighted_score = float(np.mean(weighted_scores))
            
            return {
                "relaxed_acc_keywords": relaxed_acc_keywords_score,
                "relaxed_acc_keywords_scores": relaxed_acc_keywords_scores,
                "bleu": bleu_score,
                "bleu_scores": bleu_scores,
                "weighted": weighted_score,
                "weighted_score": weighted_score,
                "weighted_scores": weighted_scores,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "total_samples": len(preds)
            }
        else:
            raise ValueError(f"不支持的指标类型: {metric_type}")