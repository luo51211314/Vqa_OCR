from loaders import VqaDataset
import editdistance
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import io
import os

class Dataset(VqaDataset):
    name = "docvqa"

    def __init__(self, split="test", **_):
        super().__init__(split)
        # 1. 本地 parquet 文件夹路径
        self.data_dir = f"/root/autodl-tmp/dataset/docVQA/data"
        # 2. 列出所有 split 对应分片
        parquet_files = sorted([f for f in os.listdir(self.data_dir)
                                if f.startswith(split) and f.endswith(".parquet")])
        if not parquet_files:
            raise FileNotFoundError(f"在 {self.data_dir} 未找到 {split} 的 parquet 文件")
        # 3. 用 pandas 顺序读取所有分片
        dfs = [pd.read_parquet(os.path.join(self.data_dir, f)) for f in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img = self._load_image(sample["image"])
        prompt = f"{sample['question']} 请直接输出答案，不要多余解释。"
        answers = sample["answers"]          # List[str]
        return img, prompt, answers, {"questionId": sample["questionId"]}
    
    def _load_image(self, image_field):
    #兼容 bytes / path / PIL.Image"""
        if isinstance(image_field, dict) and "bytes" in image_field:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if isinstance(image_field, str):
            return Image.open(image_field).convert("RGB")
        if isinstance(image_field, Image.Image):
            return image_field.convert("RGB")
        raise ValueError(f"无法识别的 image 字段类型: {type(image_field)}")

    @staticmethod
    def metrics(preds, refs, threshold=0.5):
        """
        preds: List[str]          模型预测
        refs : List[List[str]]    每题多个标准答案
        return {"anls": float}
        """
        scores = []
        for p, gts in zip(preds, refs):
            p = p.strip()
            if not p:                       # 空预测直接 0
                scores.append(0.0)
                continue
            # 1. 计算每条 gt 的归一化编辑距离相似度
            max_sim = 0.0
            for g in gts:
                g = g.strip()
                if not g:
                    continue
                dist = editdistance.eval(p.lower(), g.lower())
                max_len = max(len(p), len(g))
                sim = 1.0 - dist / max_len if max_len else 0.0
                max_sim = max(max_sim, sim)
            # 2. 阈值截断
            if max_sim < threshold:
                max_sim = 0.0
            scores.append(max_sim)

        return {"anls": float(np.mean(scores))}