from loaders import VqaDataset
import pandas as pd
import os
from PIL import Image
import io
import re
import numpy as np

class Dataset(VqaDataset):
    name = "chartqa"

    def __init__(self, split="val", **_):
        super().__init__(split)
        # ChartQA数据目录
        self.data_dir = "/root/autodl-tmp/dataset/chartQA/data"
        # 根据split选择对应的parquet文件
        if split == "train":
            parquet_files = [
                "train-00000-of-00003-49492f364babfa44.parquet",
                "train-00001-of-00003-7302bae5e425bbc7.parquet",
                "train-00002-of-00003-194c9400785577a2.parquet"
            ]
        elif split == "val":
            parquet_files = ["val-00000-of-00001-0f11003c77497969.parquet"]
        elif split == "test":
            parquet_files = ["test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"]
        else:
            raise ValueError(f"不支持的split: {split}")
        
        # 读取所有parquet文件
        dfs = []
        for file in parquet_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                dfs.append(pd.read_parquet(file_path))
        
        if not dfs:
            raise FileNotFoundError(f"在 {self.data_dir} 未找到 {split} 的parquet文件")
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        # 检查数据列名
        print(f"ChartQA数据集列名: {list(self.df.columns)}")
        print(f"前几行数据示例:")
        print(self.df.head(2))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # 加载图像 - 列名可能是 'image'
        img = self._load_image(sample["image"])
        
        # 构建提示 - 使用 'query' 而不是 'question'
        prompt = f"{sample['query']} 请直接输出答案，不要多余解释。"
        
        # 获取答案 - 使用 'label' 而不是 'answers'
        # 处理numpy数组：将numpy.ndarray转换为字符串列表
        answers = sample["label"]
        if hasattr(answers, 'dtype'):  # 如果是numpy数组
            answers = [str(gt).strip() for gt in answers]
        elif not isinstance(answers, list):
            answers = [str(answers).strip()]
        
        return img, prompt, answers, {"questionId": idx}
    
    def _load_image(self, image_field):
        """兼容 bytes / path / PIL.Image"""
        if isinstance(image_field, dict) and "bytes" in image_field:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if isinstance(image_field, str):
            return Image.open(image_field).convert("RGB")
        if isinstance(image_field, Image.Image):
            return image_field.convert("RGB")
        raise ValueError(f"无法识别的image字段类型: {type(image_field)}")

    @staticmethod
    def _is_numeric(value):
        """检查字符串是否为数字"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _relaxed_accuracy(pred, refs):
        """Relaxed accuracy: 预测包含参考答案中的任意字符即视为正确"""
        pred = str(pred).strip().lower()
        if not pred:
            return 0.0
        
        for ref in refs:
            ref = str(ref).strip().lower()
            if not ref:
                continue
            # 检查预测是否包含参考答案中的任意字符
            if any(char in pred for char in ref):
                return 1.0
        return 0.0

    @staticmethod
    def _relaxed_accuracy_numeric(pred, true):
        """计算松弛准确率：误差在真实值的5%以内即认为正确"""
        try:
            pred_num = float(pred)
            true_num = float(true)
            if true_num == 0:
                return pred_num == 0  # 避免除以零
            relative_error = abs(pred_num - true_num) / abs(true_num)
            return relative_error <= 0.05
        except (ValueError, TypeError):
            return False

    @staticmethod
    def metrics(preds, refs, metric_type="anls"):
        """
        支持多种评估指标
        metric_type: "anls", "relaxed_accuracy", "relaxed_accuracy_80"
        """
        import numpy as np

        if metric_type == "anls":
            return Dataset._metrics_anls(preds, refs)
        elif metric_type == "relaxed_accuracy":
            return Dataset._metrics_relaxed_accuracy(preds, refs)
        elif metric_type == "relaxed_accuracy_80":
            return Dataset._metrics_relaxed_accuracy_80(preds, refs)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
    
    @staticmethod
    def _metrics_anls(preds, refs):
        """ANLS指标"""
        import editdistance
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
            "total_samples": len(preds)
        }
    
    @staticmethod
    def _metrics_relaxed_accuracy(preds, refs):
        """宽松准确率：包含任意字符即正确"""
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
                
                # 检查预测是否包含参考答案中的任意字符
                if any(char in pred_str for char in ref_str):
                    max_score = 1.0
                    break
            
            relaxed_scores.append(max_score)
        
        return {
            "relaxed_accuracy": float(np.mean(relaxed_scores)),
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
            "total_samples": len(preds)
        }

    @staticmethod
    def metrics(preds, refs):
        """
        ChartQA混合评估指标：
        - 数字答案：松弛准确率（误差≤5%）
        - 文本答案：relaxed accuracy（预测包含参考答案字符即正确）
        
        preds: List[str]          模型预测
        refs : List[List[str]]    每题多个标准答案
        return {"accuracy": float, "numeric_accuracy": float, "text_accuracy": float, "relaxed_text_acc": float}
        """
        total_correct = 0
        numeric_correct = 0
        numeric_count = 0
        text_correct = 0
        text_count = 0
        relaxed_text_correct = 0
        
        for pred, gt_list in zip(preds, refs):
            pred = str(pred).strip() if pred is not None else ""
            
            # 处理numpy数组：将numpy.ndarray转换为字符串列表
            if hasattr(gt_list, 'dtype'):  # 如果是numpy数组
                gt_list = [str(gt).strip() for gt in gt_list]
            elif not isinstance(gt_list, list):
                gt_list = [str(gt_list).strip()]
            
            # 检查是否为数字答案
            is_numeric_gt = any(Dataset._is_numeric(gt) for gt in gt_list)
            
            if is_numeric_gt:
                numeric_count += 1
                # 数字答案：使用松弛准确率
                correct = any(Dataset._relaxed_accuracy_numeric(pred, gt) for gt in gt_list)
                if correct:
                    numeric_correct += 1
                    total_correct += 1
            else:
                text_count += 1
                # 文本答案：使用relaxed accuracy（预测包含参考答案字符即正确）
                relaxed_correct = any(Dataset._relaxed_accuracy(pred, [gt]) for gt in gt_list)
                if relaxed_correct:
                    relaxed_text_correct += 1
                    total_correct += 1
                    
                # 原始精确匹配（不区分大小写）
                pred_lower = pred.lower()
                exact_correct = any(str(gt).lower() == pred_lower for gt in gt_list)
                if exact_correct:
                    text_correct += 1
        
        total_samples = len(preds)
        results = {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "numeric_accuracy": numeric_correct / numeric_count if numeric_count > 0 else 0.0,
            "text_accuracy": text_correct / text_count if text_count > 0 else 0.0,
            "relaxed_text_acc": relaxed_text_correct / text_count if text_count > 0 else 0.0,
            "numeric_samples": numeric_count,
            "text_samples": text_count
        }
        
        return results