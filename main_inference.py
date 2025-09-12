#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 VQA 批推理脚本（LLaVA）
支持插件式数据集（scienceqa / docvqa / gqa / …）
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

# -------------- LLaVA 必备 import --------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" #拉clip权重

sys.path.append("/root/autodl-tmp/model/LLaVA")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# -------------- 统一数据集入口 --------------
from load_dataset import build_dataloader


# -------------- 推理配置 --------------
conv_mode = "llava_v1"
temperature = 0.1
top_p = 0.7
max_new_tokens = 128
num_beams = 4


# -------------- 主推理函数 --------------
def main():
    parser = argparse.ArgumentParser(description="LLaVA 通用 VQA 评测")
    parser.add_argument("--dataset", choices=["scienceqa", "docvqa", "gqa"], required=True)
    parser.add_argument("--split", default="validation", help="validation / test")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/llava_hug")
    parser.add_argument("--num_samples", type=int, default=None, help="仅调试：限制样本数")
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
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        device="cuda",
    )
    device = torch.device("cuda")
    model.to(device)

    # ---- 3. 批量推理 ----
    preds, refs = [], []
    sample_metas = []  # 存 question_id / idx 等额外信息
    questions = []
    start = time.time()

    for imgs, prompts, answers, extras in tqdm(loader, desc=f"{args.dataset}-{args.split}"):
        #print(prompts, answers, extras, end="\n")
        # 3.1 图像 tensor
        image_tensor = process_images(imgs, image_processor, model.config).to(
            device, dtype=torch.float16
        )
        image_sizes = [img.size for img in imgs]

        # 3.2 prompt 构造
        use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if use_im_start_end
            else DEFAULT_IMAGE_TOKEN
        )
        prompts_in = []
        for q in prompts:
            qs = image_token_se + "\n" + q
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompts_in.append(conv.get_prompt())

        # 3.3 tokenize
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                for p in prompts_in
            ],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # 3.4 generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        # 3.5 decode
        batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_preds = [p.strip() for p in batch_preds]

        # 3.6 收集
        #print("preds:",batch_preds)
        preds.extend(batch_preds)
        refs.extend(answers)
        sample_metas.extend(extras)
        questions.extend([q.split("\n")[0] for q in prompts])

    elapsed = time.time() - start

    # ---- 4. 指标 ----
    module = importlib.import_module(f"loaders.{args.dataset}")
    metrics = module.Dataset.metrics(preds, refs)
    print("\n=== 评测结果 ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---- 5. 落盘 ----
    os.makedirs("results", exist_ok=True)
    basename = f"{args.dataset}_{args.split}"
    # 5.1 详细 csv
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

    # 5.2 指标 json
    metrics["processing_time"] = round(elapsed, 2)
    json.dump(metrics, open(f"results/{basename}_metrics.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

    print(f"\n结果已保存到 results/{basename}_*")


if __name__ == "__main__":
    main()