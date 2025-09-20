#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 VQA 批推理脚本（支持多种模型和指标）
支持插件式数据集（scienceqa / docvqa / gqa / chartqa）
支持多种模型（llava / qwen）
支持多种指标（anls / relaxed_accuracy / relaxed_accuracy_80）
支持专家模块增强
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

# -------------- 专家模块 --------------
from choose_expert import ExpertChooser
from expert.expert_manager import ExpertManager

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
    # 专家模块参数
    parser.add_argument("--use_experts", choices=["auto", "manual", "off"], default="off", 
                       help="专家模块使用模式: auto-自动选择, manual-手动指定, off-禁用")
    parser.add_argument("--expert_names", default="", 
                       help="手动模式下的专家名称列表，逗号分隔，如: text,chart")
    args = parser.parse_args()

    # ---- 0. 专家模块初始化 ----
    active_experts = []
    expert_suffix = ""
    expert_manager = None
    
    if args.use_experts != "off":
        if args.use_experts == "auto":
            # 自动选择专家
            all_experts = ExpertChooser.choose_experts_for_dataset(args.dataset)
            # 只使用第一个专家模块
            active_experts = [all_experts[0]] if all_experts else []
            print(f"自动选择专家模块: {active_experts} (只使用第一个专家)")
        elif args.use_experts == "manual" and args.expert_names:
            # 手动指定专家，只使用第一个
            all_experts = [expert.strip() for expert in args.expert_names.split(",") if expert.strip()]
            active_experts = [all_experts[0]] if all_experts else []
            print(f"手动指定专家模块: {active_experts} (只使用第一个专家)")
        
        # 初始化专家管理器
        if active_experts:
            expert_manager = ExpertManager()
            expert_name = active_experts[0]
            
            try:
                # 获取专家配置
                expert_config = ExpertChooser.get_expert_config(expert_name)
                
                # 初始化专家模块
                expert_manager.initialize_expert(expert_name, **expert_config)
                print(f"成功初始化专家模块: {expert_name}")
                
                # 生成专家后缀用于文件名
                expert_suffix = f"_{expert_name}"
                print(f"使用专家模块增强: {expert_name}")
                
            except Exception as e:
                print(f"专家模块 {expert_name} 初始化失败: {e}")
                print("将使用原始推理流程")
                active_experts = []
                expert_manager = None
        else:
            print("未找到合适的专家模块，将使用原始推理流程")
    else:
        print("专家模块已禁用，使用原始推理流程")

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
    enhanced_prompts = []  # 保存专家增强后的prompt
    start = time.time()

    for imgs, prompts, answers, extras in tqdm(loader, desc=f"{args.dataset}-{args.split}-{args.metric_type}"):
        batch_preds = []
        batch_processed_prompts = []  # 保存处理后的prompt
        batch_enhanced_prompts = []   # 保存专家增强后的prompt
        
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
        
            # 4.2 prompt处理 - 专家模块增强
            original_prompt = prompts[i]
            enhanced_prompt = original_prompt
            
            # 如果有活跃的专家模块，使用专家模块进行真正的prompt增强
            if active_experts and expert_manager:
                try:
                    # 使用专家管理器进行prompt增强
                    enhanced_prompt = expert_manager.process_with_experts(
                        imgs[i], original_prompt, active_experts
                    )
                    # print(f"专家增强后的prompt长度: {len(enhanced_prompt)}")
                    # 为了调试，显示部分增强后的prompt
                    # if len(enhanced_prompt) > 100:
                    #     print(f"增强prompt预览: {enhanced_prompt[:100]}...")
                    # else:
                    #     print(f"增强prompt: {enhanced_prompt}")
                except Exception as e:
                    print(f"专家模块处理失败: {e}")
                    enhanced_prompt = original_prompt
            
            prompt_in = model_loader.process_prompt(enhanced_prompt, args.metric_type)
            batch_processed_prompts.append(prompt_in)
            batch_enhanced_prompts.append(enhanced_prompt)
            
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
        questions.extend(batch_processed_prompts)
        enhanced_prompts.extend(batch_enhanced_prompts)

    elapsed = time.time() - start

    # ---- 5. 指标 ----
    module = importlib.import_module(f"loaders.{args.dataset}")
    metrics = module.Dataset.metrics(preds, refs, metric_type=args.metric_type)
    print("\n=== 评测结果 ===")
    print(f"{args.metric_type}: {metrics[args.metric_type]}")
    print(f"total_samples: {metrics['total_samples']}")
    
    # 添加专家模块使用信息到指标
    metrics["use_experts"] = args.use_experts
    metrics["expert_names"] = active_experts
    
    # ---- 6. 落盘 ----
    os.makedirs("results", exist_ok=True)
    basename = f"{args.dataset}_{args.split}_{args.metric_type}{expert_suffix}"
    
    # 6.1 详细 csv
    import pandas as pd
    # 获取每条数据的分数
    score_key = f"{args.metric_type}_scores"
    scores = metrics.get(score_key, [0.0] * len(preds))
    
    detail_df = pd.DataFrame(
        {
            "question_id": [m.get("questionId", m.get("sample_id", idx)) for idx, m in enumerate(sample_metas)],
            "question": questions,
            "enhanced_question": enhanced_prompts,  # 添加专家增强后的prompt
            "predicted_answer": preds,
            "ground_truth": refs,
            "score": scores,
        }
    )
    detail_df.to_csv(f"results/{basename}_detail.csv", index=False)
    
    # 6.2 指标 json
    json_metrics = {
        args.metric_type: metrics[args.metric_type],
        "total_samples": metrics["total_samples"],
        "processing_time": round(elapsed, 2),
        "model_type": args.model_type,
        "metric_type": args.metric_type,
        "use_experts": args.use_experts,
        "expert_names": active_experts
    }
    json.dump(json_metrics, open(f"results/{basename}_metrics.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到 results/{basename}_*_expert")


if __name__ == "__main__":
    main()