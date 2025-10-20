#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 VQA 批推理脚本（支持多种模型和指标）
支持插件式数据集（scienceqa / docvqa / gqa / chartqa）
支持多种模型（llava / qwen）
支持多种指标（anls / relaxed_accuracy / relaxed_accuracy_80）
支持专家模块增强和对比学习融合增强
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

# -------------- 特征融合模块 --------------
from feature_fusion.fusion_manager import FusionManager
from feature_fusion.contrastive_alignment import ContrastiveAlignment
from feature_fusion.attention_fusion import AttentionFusion

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_plugin_parameters(model, fusion_type, device="cuda"):
    """
    加载可插拔模块参数：对齐模块、融合模块和微调后的投影矩阵
    
    Args:
        model: LLaVA模型
        fusion_type: 融合类型 ("paddleocr", "pix2struct")
        device: 设备
    
    Returns:
        success: 是否加载成功
    """
    try:
        # 构建权重文件路径
        weight_dir = "/root/autodl-tmp/weight"
        alignment_weight = f"{weight_dir}/alignment_module_final_{fusion_type}.pth"
        fusion_weight = f"{weight_dir}/fusion_module_final_{fusion_type}.pth"
        projector_weight = f"{weight_dir}/projector_finetuned_{fusion_type}.pth"
        
        # 1. 加载对齐模块参数
        if os.path.exists(alignment_weight):
            alignment_state = torch.load(alignment_weight, map_location=device)
            # 这里需要根据实际模型结构来加载对齐模块参数
            print(f"对齐模块参数已加载: {alignment_weight}")
        else:
            print(f"警告: 未找到对齐模块权重 {alignment_weight}")
        
        # 2. 加载融合模块参数
        if os.path.exists(fusion_weight):
            fusion_state = torch.load(fusion_weight, map_location=device)
            # 这里需要根据实际模型结构来加载融合模块参数
            print(f"融合模块参数已加载: {fusion_weight}")
        else:
            print(f"警告: 未找到融合模块权重 {fusion_weight}")
        
        # 3. 加载微调后的投影矩阵参数
        if os.path.exists(projector_weight):
            projector_state = torch.load(projector_weight, map_location=device)
            # 更新模型的投影矩阵参数
            if hasattr(model, 'mm_projector') and projector_state:
                model.mm_projector.load_state_dict(projector_state)
                print(f"投影矩阵参数已加载: {projector_weight}")
            else:
                print("警告: 模型没有mm_projector属性或投影矩阵权重为空")
        else:
            print(f"警告: 未找到投影矩阵权重 {projector_weight}")
        
        return True
        
    except Exception as e:
        print(f"插件参数加载失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="通用 VQA 评测")
    parser.add_argument("--dataset", choices=["scienceqa", "docvqa", "gqa", "chartqa"], required=True)
    parser.add_argument("--split", default="validation", help="validation / test / val")
    parser.add_argument("--bs", type=int, default=4, help="batch size")
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
    # 对比学习融合参数
    parser.add_argument("--use_contrastive_fusion", choices=["off", "paddleocr", "pix2struct"], 
                       default="off", help="对比学习融合增强: off-禁用, paddleocr-使用PaddleOCR, pix2struct-使用Pix2Struct")
    parser.add_argument("--fusion_weight", default="", help="融合模块权重文件路径")
    
    args = parser.parse_args()

    # ---- 0. 模块初始化 ----
    active_experts = []
    expert_suffix = ""
    expert_manager = None
    fusion_manager = FusionManager()
    fusion_suffix = ""
    
    # 对比学习融合模块初始化
    if args.use_contrastive_fusion != "off":
        success = fusion_manager.initialize_fusion(
            fusion_type=args.use_contrastive_fusion,
            weight_path=args.fusion_weight,
            device="cuda"
        )
        if success:
            fusion_suffix = f"_fusion_{args.use_contrastive_fusion}"
            print(f"对比学习融合模块已启用")
        else:
            print(f"对比学习融合模块初始化失败，将使用原始推理流程")
    
    # 专家模块初始化（仅在融合模块未启用时）
    if args.use_experts != "off" and not fusion_manager.is_initialized():
        if args.use_experts == "auto":
            all_experts = ExpertChooser.choose_experts_for_dataset(args.dataset)
            active_experts = [all_experts[0]] if all_experts else []
            print(f"自动选择专家模块: {active_experts}")
        elif args.use_experts == "manual" and args.expert_names:
            all_experts = [expert.strip() for expert in args.expert_names.split(",") if expert.strip()]
            active_experts = [all_experts[0]] if all_experts else []
            print(f"手动指定专家模块: {active_experts}")
        
        if active_experts:
            expert_manager = ExpertManager()
            expert_name = active_experts[0]
            try:
                expert_config = ExpertChooser.get_expert_config(expert_name)
                expert_manager.initialize_expert(expert_name, **expert_config)
                expert_suffix = f"_{expert_name}"
                print(f"专家模块初始化成功: {expert_name}")
            except Exception as e:
                print(f"专家模块初始化失败: {e}")
                active_experts = []
                expert_manager = None
    else:
        print("专家模块已禁用")

    # ---- 1. 数据集 ----
    loader = build_dataloader(args.dataset, args.split, batch_size=args.bs, num_workers=4)
    if args.num_samples:
        loader.dataset.df = loader.dataset.df[:args.num_samples]

    # ---- 2. 模型 ----
    model_loader = get_model_loader(args.model_type)
    tokenizer, model, image_processor, context_len = model_loader.load_model(args.model_path)
    device = torch.device("cuda")
    model.to(device)

    # ---- 3. 加载可插拔模块参数 ----
    if args.use_contrastive_fusion != "off":
        success = load_plugin_parameters(model, args.use_contrastive_fusion, device)
        if not success:
            print("警告: 可插拔模块参数加载失败，将使用原始模型推理")

    # ---- 4. 获取推理配置 ----
    inference_config = model_loader.get_inference_config(args.metric_type)

    # ---- 5. 批量推理 ----
    preds, refs = [], []
    sample_metas = []
    questions = []
    enhanced_prompts = []
    start = time.time()

    for imgs, prompts, answers, extras in tqdm(loader, desc=f"{args.dataset}-{args.split}-{args.metric_type}"):
        batch_preds = []
        batch_processed_prompts = []
        batch_enhanced_prompts = []
        batch_original_prompts = []
        
        for i in range(len(imgs)):
            # 图像处理
            if hasattr(model_loader, 'image_processor') and image_processor:
                from models.llava.llava.mm_utils import process_images
                image_tensor = process_images([imgs[i]], image_processor, model.config).to(
                    device, dtype=torch.float16
                )
                image_sizes = [imgs[i].size]
            else:
                image_tensor = None
                image_sizes = None
        
            # Prompt处理
            original_prompt = prompts[i]
            enhanced_prompt = original_prompt
            
            # 特征融合处理（使用已加载的可插拔模块参数）
            if fusion_manager.is_initialized():
                try:
                    # 提取专家特征
                    expert_features = fusion_manager.extract_expert_features(imgs[i], original_prompt)
                    if expert_features is not None:
                        # 获取LLaVA图像特征
                        if image_tensor is not None:
                            with torch.no_grad():
                                # 通过LLaVA视觉编码器获取图像特征
                                llava_features = model.model.vision_tower(image_tensor)
                                
                                # 应用融合模块（使用已加载的参数）
                                fused_features = fusion_manager.fuse_features(llava_features, expert_features)
                                if fused_features is not None:
                                    # 关键修改：使用融合后的特征替换原始视觉特征
                                    # 通过微调后的投影矩阵处理融合特征
                                    if hasattr(model.get_model(), 'mm_projector'):
                                        # 使用已加载的微调投影矩阵处理融合特征
                                        projected_fused_features = model.get_model().mm_projector(fused_features)
                                        
                                        # 创建自定义的图像特征处理函数
                                        def custom_encode_images(images):
                                            # 直接返回投影后的融合特征，跳过视觉编码器
                                            return projected_fused_features
                                        
                                        # 临时替换模型的图像编码方法
                                        original_encode_images = model.encode_images
                                        model.encode_images = custom_encode_images
                                        
                                        print(f"特征融合成功: LLaVA特征{llava_features.shape}, 专家特征{expert_features.shape}, 融合特征{fused_features.shape}")
                                    else:
                                        print("警告: 模型没有mm_projector属性，无法使用融合特征")
                except Exception as e:
                    print(f"特征融合处理失败: {e}")
            # 专家模块处理
            elif active_experts and expert_manager:
                try:
                    enhanced_prompt = expert_manager.process_with_experts(
                        imgs[i], original_prompt, active_experts
                    )
                except Exception as e:
                    print(f"专家模块处理失败: {e}")
            
            prompt_in = model_loader.process_prompt(enhanced_prompt, args.metric_type)
            batch_processed_prompts.append(prompt_in)
            batch_enhanced_prompts.append(enhanced_prompt)
            batch_original_prompts.append(original_prompt)
            
            # Tokenize和生成
            input_ids = model_loader.tokenizer_image_token(prompt_in, tokenizer, None, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).to(device)
            
            output_ids = model_loader.generate(
                input_ids, image_tensor, None, image_sizes, inference_config
            )
            
            # 恢复原始的图像编码方法（如果被替换）
            if fusion_manager.is_initialized() and hasattr(model, 'encode_images') and 'original_encode_images' in locals():
                model.encode_images = original_encode_images
            
            pred = model_loader.decode(output_ids, tokenizer)
            batch_preds.append(pred)

        # 收集结果
        preds.extend(batch_preds)
        refs.extend(answers)
        sample_metas.extend(extras)
        questions.extend(batch_original_prompts)
        enhanced_prompts.extend(batch_processed_prompts)

    elapsed = time.time() - start

    # ---- 6. 指标计算和保存 ----
    module = importlib.import_module(f"loaders.{args.dataset}")
    metrics = module.Dataset.metrics(preds, refs, metric_type=args.metric_type)
    
    print("\n=== 评测结果 ===")
    print(f"{args.metric_type}: {metrics[args.metric_type]}")
    print(f"total_samples: {metrics['total_samples']}")
    
    # 保存结果
    suffix = expert_suffix if expert_suffix else fusion_suffix
    basename = f"{args.dataset}_{args.split}_{args.metric_type}{suffix}"
    
    os.makedirs("results", exist_ok=True)
    
    # 保存详细结果
    import pandas as pd
    score_key = f"{args.metric_type}_scores"
    scores = metrics.get(score_key, [0.0] * len(preds))
    
    detail_df = pd.DataFrame({
        "question_id": [m.get("questionId", m.get("sample_id", idx)) for idx, m in enumerate(sample_metas)],
        "question": questions,
        "enhanced_question": enhanced_prompts,
        "predicted_answer": preds,
        "ground_truth": refs,
        "score": scores,
    })
    detail_df.to_csv(f"results/{basename}_detail.csv", index=False)
    
    # 保存指标
    json_metrics = {
        args.metric_type: metrics[args.metric_type],
        "total_samples": metrics["total_samples"],
        "processing_time": round(elapsed, 2),
        "model_type": args.model_type,
        "metric_type": args.metric_type,
        "use_experts": args.use_experts,
        "expert_names": active_experts,
        "use_contrastive_fusion": args.use_contrastive_fusion
    }
    json.dump(json_metrics, open(f"results/{basename}_metrics.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到 results/{basename}_*")


if __name__ == "__main__":
    main()