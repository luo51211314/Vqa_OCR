#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成固定 dev 子集 id 文件（模板消融专用，seed=42 锁定）。

    python scripts/make_dev_subset.py
    -> experiments/subsets/chartqa_test_dev300.ids
       experiments/subsets/docvqa_val_dev300.ids

纪律：P1/P2/P3 模板只允许在该子集上比较；子集 id 落盘并提交，
保证"模板选择不接触测试集结果"可审计。

注意：ChartQA 的 parquet 没有 question id 列，使用 DataFrame 行索引作为稳定 id
（加载顺序由 parquet 文件列表确定，可复现；与 detail CSV 中 question_id 的
idx 回退口径一致）。
"""
import importlib
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SUBSET_DIR = os.path.join(REPO_ROOT, "experiments", "subsets")
ID_COLS = ("question_id", "questionId", "id", "sample_id")

# (dataset, split, n) —— 与 experiments/configs/ablation_template.yaml 对应
SPECS = [
    ("chartqa", "test", 300),
    ("docvqa", "val", 300),
]


def main():
    os.makedirs(SUBSET_DIR, exist_ok=True)

    for dataset, split, n in SPECS:
        module = importlib.import_module(f"loaders.{dataset}")
        ds = module.Dataset(split=split)
        df = ds.df
        id_col = next((c for c in ID_COLS if c in df.columns), None)

        rng = np.random.RandomState(42)
        chosen = sorted(rng.choice(len(df), size=min(n, len(df)), replace=False))
        if id_col is not None:
            ids = df.iloc[chosen][id_col].astype(str).tolist()
            id_kind = id_col
        else:
            # 无 id 列（ChartQA）：用行索引，main_inference 识别 __row_index__
            ids = [str(j) for j in chosen]
            id_kind = "__row_index__"

        out = os.path.join(SUBSET_DIR, f"{dataset}_{split}_dev{n}.ids")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        print(f"{dataset}/{split}: {len(df)} -> 抽取 {len(ids)} 个 id "
              f"(id={id_kind}) -> {out}")


if __name__ == "__main__":
    main()
