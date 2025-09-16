#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 VQA 批推理脚本（支持多种模型和指标）
支持插件式数据集（scienceqa / docvqa / gqa / chartqa）
支持多种模型（llava / qwen）
支持多种指标（anls / relaxed_accuracy / relaxed_accuracy_80）
"""

import os
import sys
import time
import json
import argparse
import importlib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# -------------- 模型加载器 --------------
from model_loader import get_model_loader

# -------------- 统一数据集入口 --------------
from load_dataset import build_dataloader


# -------------- 主推理函数 --------------
def main():
    parser = argparse.ArgumentParser(description="通用 VQA 评测")
    parser.add_argument("--dataset", choices=["scienceqa", "docvqa", "gqa", "chartqa"], required=True)
    parser.add_argument("--split", default="validation", help="validation / test / val")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--num_samples", type=int, default=None, help="仅调试：限制样本数")
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/llava_hug")
    parser.add_argument("--model_type", choices=["llava", "qwen"], default="llava", help="模型类型")
    parser.add_argument("--metric_type", choices=["anls", "relaxed_accuracy", "relaxed_accuracy_80"], 
                       default="anls", help="评估指标类型")
    args = parser.parse_args()

    # ---- 1. 数据集 ----
    loader = build_dataloader(
        args.dataset,
        args.split,
        batch_size=args.bs,
        num_workers=4,
    )
    if args.num_samples:  # 调试用
        loader.dataset.df = loader.dataset.df[:args.num_samples]

    # ---- 2. 模型 ----
    model_loader = get_model_loader(args.model_type)
    tokenizer, model, image_processor, context_len = model_loader.load_model(args.model_path)
    device = torch.device("cuda")
    model.to(device)

    # ---- 3. 获取推理配置 ----
    inference_config = model_loader.get_inference_config(args.metric_type)
    temperature = inference_config["temperature"]
    top_p = inference_config["top_p"]
    max_new_tokens = inference_config["max_new_tokens"]
    num_beams = inference_config["num_beams"]

    # ---- 4. 批量推理 ----
    preds, refs = [], []
    sample_metas = []
    questions = []
    start = time.time()

    for imgs, prompts, answers, extras in tqdm(loader, desc=f"{args.dataset}-{args.split}-{args.metric_type}"):
        batch_preds = []
        
        # 逐个处理batch中的样本
        for i in range(len(imgs)):
            # 4.1 图像处理
            if hasattr(model_loader, 'image_processor') and image_processor:
                from models.llava.llava.mm_utils import process_images
                image_tensor = process_images([imgs[i]], image_processor, model.config).to(
                    device, dtype=torch.float16
                )
                image_sizes = [imgs[i].size]
            else:
                # 对于不支持图像处理的模型（如纯文本模型）
                image_tensor = None
                image_sizes = None

            # 4.2 prompt处理
            prompt_in = model_loader.process_prompt(prompts[i], args.metric_type)
            
            # 4.3 tokenize
            input_ids = model_loader.tokenizer_image_token(prompt_in, tokenizer, None, return_tensors="pt")
            
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = None

            # ---------- generate ----------
            output_ids = model_loader.generate(
                input_ids, image_tensor, attention_mask, image_sizes, inference_config
            )

            # 4.5 decode
            pred = model_loader.decode(output_ids, tokenizer)
            batch_preds.append(pred)

        # 4.6 收集
        preds.extend(batch_preds)
        refs.extend(answers)
        sample_metas.extend(extras)
        questions.extend([prompts[i].split("\n")[0] for i in range(len(prompts))])

    elapsed = time.time() - start

    # ---- 5. 指标 ----
    module = importlib.import_module(f"loaders.{args.dataset}")
    metrics = module.Dataset.metrics(preds, refs, metric_type=args.metric_type)
    print("\n=== 评测结果 ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---- 6. 落盘 ----
    os.makedirs("results", exist_ok=True)
    basename = f"{args.dataset}_{args.split}_{args.metric_type}"
    
    # 6.1 详细 csv
    import pandas as pd
    detail_df = pd.DataFrame(
        {
            "question_id": [m.get("questionId", m.get("sample_id", idx)) for idx, m in enumerate(sample_metas)],
            "question": questions,
            "predicted_answer": preds,
            "ground_truth": refs,
        }
    )
    detail_df.to_csv(f"results/{basename}_detail.csv", index=False)

    # 6.2 指标 json
    metrics["processing_time"] = round(elapsed, 2)
    metrics["model_type"] = args.model_type
    metrics["metric_type"] = args.metric_type
    json.dump(metrics, open(f"results/{basename}_metrics.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

    print(f"\n结果已保存到 results/{basename}_*")


if __name__ == "__main__":
    main()