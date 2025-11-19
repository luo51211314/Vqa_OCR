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

# -------------- 统一数据集入口 --------------
from load_dataset import build_dataloader

# -------------- 专家模块 --------------
from choose_expert import ExpertChooser
from expert.expert_manager import ExpertManager

# -------------- 导入LLaVAOCRModel --------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_fusion'))
from train_fusion.llava_ocr_model import LLaVAOCRModel
from transformers import BitsAndBytesConfig

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置环境变量，处理CLIP权重缓存
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"

def update_model_params(target_component, param_dict, prefix_to_remove, component_name, training_keys=None, nested_key=None):
    """
    更新模型组件中的参数，支持索引形式的参数访问
    
    Args:
        target_component: 目标模型组件（如mm_projector）
        param_dict: 包含参数的字典
        prefix_to_remove: 需要从参数名中移除的前缀
        component_name: 组件名称（用于日志输出）
        training_keys: 需要跳过的训练相关参数列表
        nested_key: 如果参数在嵌套字典中，指定嵌套键名
    
    Returns:
        tuple: (updated_keys, skipped_keys)
    """
    updated_keys = []
    skipped_keys = []
    
    # 处理嵌套字典的情况
    if nested_key and nested_key in param_dict and isinstance(param_dict[nested_key], dict):
        params_to_process = param_dict[nested_key].items()
    else:
        # 处理非嵌套字典的情况，过滤掉训练参数
        params_to_process = []
        for key, value in param_dict.items():
            if training_keys and key in training_keys:
                print(f"跳过训练参数: {key}")
                skipped_keys.append(key)
                continue
            params_to_process.append((key, value))
    
    # 处理每个参数
    for param_key, param_value in params_to_process:
        # 去掉指定前缀
        clean_param_key = param_key.replace(prefix_to_remove, '')
        
        # 尝试处理索引形式的参数（如0.weight, 2.bias）
        try:
            if '.' in clean_param_key:
                index_part, param_part = clean_param_key.split('.')
                # 尝试将索引部分转换为整数
                try:
                    index = int(index_part)
                    # 检查目标组件是否有足够的模块，且该模块有对应参数
                    if hasattr(target_component, '__getitem__') and index < len(target_component):
                        layer = target_component[index]
                        if hasattr(layer, param_part) and isinstance(getattr(layer, param_part), torch.nn.Parameter):
                            param_in_model = getattr(layer, param_part)
                            if param_in_model.shape == param_value.shape:
                                setattr(layer, param_part, torch.nn.Parameter(param_value))
                                updated_keys.append(f"{param_key} -> {component_name}[{index}].{param_part}")
                                print(f"成功更新参数: {param_key} -> {component_name}[{index}].{param_part}")
                                continue
                    # 如果不支持__getitem__，尝试直接通过属性访问
                    elif hasattr(target_component, index_part):
                        layer = getattr(target_component, index_part)
                        if hasattr(layer, param_part) and isinstance(getattr(layer, param_part), torch.nn.Parameter):
                            param_in_model = getattr(layer, param_part)
                            if param_in_model.shape == param_value.shape:
                                setattr(layer, param_part, torch.nn.Parameter(param_value))
                                updated_keys.append(f"{param_key} -> {component_name}.{index_part}.{param_part}")
                                print(f"成功更新参数: {param_key} -> {component_name}.{index_part}.{param_part}")
                                continue
                except (ValueError, IndexError):
                    pass
            
            # 尝试直接更新目标组件中的参数
            if hasattr(target_component, clean_param_key) and isinstance(getattr(target_component, clean_param_key), torch.nn.Parameter):
                if getattr(target_component, clean_param_key).shape == param_value.shape:
                    setattr(target_component, clean_param_key, torch.nn.Parameter(param_value))
                    updated_keys.append(f"{param_key} -> {component_name}.{clean_param_key}")
                    print(f"成功更新参数: {param_key} -> {component_name}.{clean_param_key}")
                    continue
            
            # 所有尝试都失败
            print(f"{component_name}中未找到参数: {clean_param_key}")
            skipped_keys.append(param_key)
        except Exception as e:
            print(f"处理参数 {param_key} 时出错: {e}")
            skipped_keys.append(param_key)
    
    return updated_keys, skipped_keys

def load_new_params(model, new_params_dict, device="cuda"):
    """
    加载新增模块参数
    
    Args:
        model: 模型实例
        new_params_dict: 新增参数字典
        device: 设备
    
    Returns:
        tuple: (updated_keys, skipped_keys)
    """
    model_state_dict = model.state_dict()
    updated_keys = []
    skipped_keys = []
    
    # 需要跳过的训练参数
    training_params = ['epoch', 'train_loss', 'val_loss']
    
    # 处理每个参数
    for key, value in new_params_dict.items():
        # 跳过训练参数
        if key in training_params:
            print(f"跳过训练参数: {key}")
            continue
        
        # 特殊处理new_params嵌套字典
        if key == 'new_params' and isinstance(value, dict):
            print(f"解析new_params嵌套字典，包含 {len(value)} 个参数")
            for nested_key, nested_value in value.items():
                # 直接使用nested_key作为候选路径（因为模型参数已经包含完整路径）
                if nested_key in model_state_dict:
                    if model_state_dict[nested_key].shape == nested_value.shape:
                        model_state_dict[nested_key] = nested_value
                        updated_keys.append(f"new_params.{nested_key} -> {nested_key}")
                        print(f"成功映射参数: new_params.{nested_key} -> {nested_key}")
                    else:
                        print(f"形状不匹配: new_params.{nested_key} -> {nested_key} (权重形状: {nested_value.shape}, 模型形状: {model_state_dict[nested_key].shape})")
                else:
                    skipped_keys.append(f"new_params.{nested_key}")
                    print(f"未找到匹配的参数路径: {nested_key}")
            continue
        
        # 对于其他参数，尝试常规匹配（直接使用key作为候选路径）
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                updated_keys.append(f"{key} -> {key}")
                print(f"成功映射参数: {key} -> {key}")
            else:
                print(f"形状不匹配: {key} -> {key} (权重形状: {value.shape}, 模型形状: {model_state_dict[key].shape})")
        else:
            skipped_keys.append(key)
            print(f"未找到匹配的参数路径: {key}")
    
    # 更新模型权重
    model.load_state_dict(model_state_dict)
    
    return updated_keys, skipped_keys

def load_model_weights(model, weight_dir, device="cuda"):
    """
    加载模型权重，按照以下顺序：
    1. 解冻参数文件 (unforzen_param.pth)
    2. 新增参数文件 (new_params.pth)
    3. LoRA参数 (peft_lora目录)
    
    Args:
        model: 已初始化的模型
        weight_dir: 权重目录路径
        device: 设备
    
    Returns:
        是否成功加载权重
    """
    success = False
    
    # 1. 尝试加载解冻参数文件
    unfrozen_path = os.path.join(weight_dir, "unfrozen_params.pth")
    if os.path.exists(unfrozen_path):
        print(f"加载解冻参数文件: {unfrozen_path}")
        try:
            unfrozen_state_dict = torch.load(unfrozen_path, map_location=device)
            
            # 跳过训练相关参数（不是模型权重）
            training_keys = ['epoch', 'train_loss', 'val_loss']
            
            # 获取llava_model实例
            if hasattr(model, 'llava_model') and hasattr(model.llava_model, 'get_model'):
                llava_model_instance = model.llava_model.get_model()
                
                # 查找解冻参数对应的目标组件（避免硬编码mm_projector）
                target_component = None
                component_name = None
                prefix_to_remove = None
                
                # 检查unfrozen_params字典中是否有mm_projector相关参数
                if 'unfrozen_params' in unfrozen_state_dict and isinstance(unfrozen_state_dict['unfrozen_params'], dict):
                    # 分析参数名，找出共同前缀
                    all_param_keys = list(unfrozen_state_dict['unfrozen_params'].keys())
                    # 尝试检测mm_projector相关的前缀
                    if any('mm_projector' in key for key in all_param_keys):
                        # 找到第一个包含mm_projector的参数名
                        first_mm_param = next(key for key in all_param_keys if 'mm_projector' in key)
                        # 提取前缀部分
                        prefix_parts = first_mm_param.split('.')[:-2]  # 移除索引和参数名部分
                        prefix_to_remove = '.'.join(prefix_parts) + '.'
                        
                        # 检查目标组件是否存在
                        if hasattr(llava_model_instance, 'mm_projector'):
                            target_component = llava_model_instance.mm_projector
                            component_name = "mm_projector"
                            print(f"成功获取{component_name}实例")
                
                # 如果找到了目标组件，更新参数
                if target_component:
                    # 处理嵌套的unfrozen_params字典
                    updated_keys, skipped_keys = update_model_params(
                        target_component=target_component,
                        param_dict=unfrozen_state_dict,
                        prefix_to_remove=prefix_to_remove,
                        component_name=component_name,
                        training_keys=training_keys,
                        nested_key='unfrozen_params'
                    )
                    
                    # 处理非嵌套的参数
                    if 'unfrozen_params' not in unfrozen_state_dict:
                        non_nested_updated, non_nested_skipped = update_model_params(
                            target_component=target_component,
                            param_dict=unfrozen_state_dict,
                            prefix_to_remove=prefix_to_remove,
                            component_name=component_name,
                            training_keys=training_keys
                        )
                        updated_keys.extend(non_nested_updated)
                        skipped_keys.extend(non_nested_skipped)
                    
                    # 输出加载结果
                    print(f"解冻参数加载完成:")
                    print(f"- 成功更新 {len(updated_keys)} 个参数")
                    print(f"- 无法映射 {len(skipped_keys)} 个参数")
                    success = len(updated_keys) > 0 or not skipped_keys
                else:
                    print(f"未找到合适的目标组件来更新解冻参数")
                    success = False
            else:
                print(f"无法获取llava_model实例")
                success = False
        except Exception as e:
            print(f"加载解冻参数失败: {e}")
    else:
        print(f"解冻参数文件不存在: {unfrozen_path}")
    
    # 2. 尝试加载新增参数文件
    new_params_path = os.path.join(weight_dir, "new_params.pth")
    if os.path.exists(new_params_path):
        print(f"加载新增参数文件: {new_params_path}")
        try:
            new_params_dict = torch.load(new_params_path, map_location=device)
            model_state_dict = model.state_dict()
            updated_keys = []
            skipped_keys = []
            
            # 调用load_new_params函数加载新增参数
            updated_keys, skipped_keys = load_new_params(model, new_params_dict, device)
            
            # 打印无法映射的参数
            if skipped_keys:
                print(f"\n无法映射的参数列表: {skipped_keys}")
            print(f"\n新增参数加载完成:")
            print(f"- 成功更新 {len(updated_keys)} 个参数")
            if skipped_keys:
                print(f"- 无法映射 {len(skipped_keys)} 个参数")
            success = success or len(updated_keys) > 0
        except Exception as e:
            print(f"加载新增参数失败: {e}")
    else:
        print(f"新增参数文件不存在: {new_params_path}")
    
    # 3. 尝试加载LoRA参数
    lora_dir = os.path.join(weight_dir, "peft_lora")
    if os.path.exists(lora_dir):
        print(f"加载LoRA参数: {lora_dir}")
        try:
            from peft import PeftModel
            # 首先检查model.llava_model是否已经是PeftModel（根据llava_ocr_model.py中的设置）
            if hasattr(model, 'llava_model'):
                # 直接对llava_model应用LoRA，因为在训练时self.llava_model已经是通过get_peft_model创建的
                if hasattr(model.llava_model, 'peft_config'):
                    # 如果llava_model已经有LoRA配置，直接加载权重
                    model.llava_model.load_adapter(lora_dir, adapter_name="default")
                    print(f"LoRA权重加载成功到model.llava_model已有的适配器")
                else:
                    # 确保llava_model.base_model.model存在，然后应用LoRA
                    if hasattr(model.llava_model, 'base_model') and hasattr(model.llava_model.base_model, 'model'):
                        model.llava_model.base_model.model = PeftModel.from_pretrained(
                            model.llava_model.base_model.model,
                            lora_dir,
                            adapter_name="default",
                            is_trainable=False
                        )
                        print(f"LoRA参数加载到llava_model.base_model.model层级成功")
                    else:
                        # 尝试直接对llava_model应用LoRA
                        model.llava_model = PeftModel.from_pretrained(
                            model.llava_model,
                            lora_dir,
                            adapter_name="default",
                            is_trainable=False
                        )
                        print(f"LoRA参数直接加载到model.llava_model层级成功")
                success = True
            else:
                # 回退到原始加载方式
                if hasattr(model, 'peft_config'):
                    # 如果已经有LoRA配置，只加载权重不创建新适配器
                    model.load_adapter(lora_dir, adapter_name="default")
                    print(f"LoRA权重通过回退方式加载成功到已有的适配器")
                else:
                    model = PeftModel.from_pretrained(model, lora_dir)
                    print(f"LoRA参数通过回退方式加载成功")
                success = True
        except Exception as e:
            print(f"加载LoRA参数失败: {e}")
    else:
        print(f"LoRA参数目录不存在: {lora_dir}")
    
    return success

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
    # 特征融合模块参数
    parser.add_argument("--use_feature_fusion", choices=["off", "fusion"], 
                       default="off", help="特征融合模块: off-禁用, fusion-使用特征融合")
    parser.add_argument("--weight_dir", default="/root/autodl-tmp/weight/stage_2/epoch_2", help="权重目录路径")
    
    args = parser.parse_args()

    # ---- 0. 模块初始化 ----
    active_experts = []
    expert_suffix = ""
    expert_manager = None
    fusion_suffix = ""
    expert_model = None  # 用于存储专家模型实例
    llava_model = None
    tokenizer = None
    
    # ---- 1. 数据集 ----
    loader = build_dataloader(args.dataset, args.split, batch_size=args.bs, num_workers=4)
    if args.num_samples:
        loader.dataset.df = loader.dataset.df[:args.num_samples]

    # ---- 2. 模型 ----
    device = torch.device("cuda")
    model = None
    
    # 动态导入model_loader，避免初始化时的导入错误
    from model_loader import get_model_loader
    
    # 统一从model_loader加载基础LLaVA模型
    print(f"加载LLaVA模型: {args.model_path}")
    model_loader = get_model_loader(args.model_type)
    tokenizer, model, image_processor, context_len = model_loader.load_model(args.model_path)
    # 将模型设置为半精度模式以支持fp16权重
    model.to(device, dtype=torch.float16)
    print("LLaVA模型加载成功（使用fp16模式）")
    
    # 初始化专家模块和特征融合模块
    if args.use_experts != "off" or args.use_feature_fusion == "fusion":
        # 无论是否启用fusion，都在这里处理专家模块逻辑
        if args.use_experts != "off":
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
                    expert_model = expert_manager.initialize_expert(expert_name, **expert_config)
                    expert_suffix = f"_{expert_name}"
                    print(f"专家模块初始化成功: {expert_name}")
                except Exception as e:
                    print(f"专家模块初始化失败: {e}")
                    active_experts = []
                    expert_manager = None
        
        # 特征融合模块初始化（放在专家模块内部）
        if args.use_feature_fusion == "fusion":
            fusion_suffix = "_fusion"
            print("启用特征融合llava_ocr_model")           
            # 从配置文件加载配置
            config_path = "/root/autodl-tmp/codes/Vqa_ocr/train_fusion/config.json"
            print(f"从配置文件加载配置: {config_path}")
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新模型路径
                config['model_config']['llava_model_path'] = args.model_path
                # # 推理时禁用LoRA
                # config['lora_config']['lora_enable'] = False
                # 推理时禁用梯度检查点
                config['training_config']['gradient_checkpointing'] = False
                
                print(f"配置加载成功，模型配置: {config['model_config']}")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                # 创建默认配置作为后备
                config = {
                    'model_config': {
                        'llava_model_path': args.model_path,
                        'ocr_model_path': "/root/autodl-tmp/model/ppocr_hug",
                        'vision_select_layer': -1,
                        'vision_select_feature': "patch",
                        'projector_type': "mlp2x"
                    },
                    'training_config': {
                        'max_length': 2048,
                        'gradient_checkpointing': False
                    },
                    'lora_config': {
                        'lora_enable': True
                    }
                }
            
            # 初始化LLaVAOCRModel
            print("初始化LLaVAOCRModel")
            llava_model = model  # 使用从model_loader加载的模型
            
            # 确保llava_model以fp16格式运行
            llava_model.to(device, dtype=torch.float16)
            
            model = LLaVAOCRModel(
                config=config,
                llava_model=llava_model,
                tokenizer=tokenizer,
                ocr_model=expert_model,
            )
            model.to(device)
            
            # 加载权重
            print(f"加载权重文件从: {args.weight_dir}")
            success = load_model_weights(model, args.weight_dir, device)
            if not success:
                print("警告: 权重加载失败，将使用原始模型推理")
            
        else:
            print("特征融合模块已禁用")
    else:
        print("专家模块和特征融合模块均已禁用")

    # ---- 3. 获取推理配置 ----
    # 无论是否使用专家模块，统一使用model_loader的推理配置
    inference_config = model_loader.get_inference_config(args.metric_type)

    # ---- 4. 批量推理 ----
    preds, refs = [], []
    sample_metas = []
    questions = []
    
    # 非fusion模式下才创建enhanced_prompts
    if args.use_feature_fusion != "fusion":
        enhanced_prompts = []
    
    start = time.time()

    for imgs, prompts, answers, extras in tqdm(loader, desc=f"{args.dataset}-{args.split}-{args.metric_type}"):
        batch_preds = []
        batch_processed_prompts = []
        batch_original_prompts = []
        
        # 非fusion模式下才创建batch_enhanced_prompts
        if args.use_feature_fusion != "fusion":
            batch_enhanced_prompts = []
        
        for i in range(len(imgs)):
            # 基础处理
            original_prompt = prompts[i]
            
            if args.use_feature_fusion == "fusion":
                # 使用LLaVAOCRModel的generate方法进行推理，通过autocast启用fp16推理
                try:
                    with torch.no_grad():
                            pred = model.generate(
                                image=imgs[i],
                                prompt=original_prompt,
                                temperature=inference_config['temperature'],
                                top_p=inference_config['top_p'],
                                max_new_tokens=inference_config['max_new_tokens']
                            )
                    batch_preds.append(pred)
                    batch_processed_prompts.append(original_prompt)
                    batch_original_prompts.append(original_prompt)
                except Exception as e:
                    print(f"融合模型推理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    batch_preds.append("Error: Failed to generate")
                    batch_processed_prompts.append(original_prompt)
                    batch_original_prompts.append(original_prompt)
            else:
                enhanced_prompt = original_prompt
                # 使用原始模型推理
                # 图像处理
                if hasattr(model_loader, 'image_processor') and image_processor:
                    from models.llava.llava.mm_utils import process_images
                    # 确保image_tensor以fp16格式输入
                    image_tensor = process_images([imgs[i]], image_processor, model.config).to(
                        device, dtype=torch.float16
                    )
                    image_sizes = [imgs[i].size]
                else:
                    image_tensor = None
                    image_sizes = None
                
                # 专家模块处理
                if active_experts and expert_manager:
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
                
                # 使用模型已设置的fp16模式进行推理
                output_ids = model_loader.generate(
                        input_ids, image_tensor, None, image_sizes, inference_config
                    )
                
                pred = model_loader.decode(output_ids, tokenizer)
                batch_preds.append(pred)

        # 收集结果
        preds.extend(batch_preds)
        refs.extend(answers)
        sample_metas.extend(extras)
        questions.extend(batch_original_prompts)
        
        # 非fusion模式下才扩展enhanced_prompts
        if args.use_feature_fusion != "fusion":
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
    
    # 根据模式决定DataFrame的列
    if args.use_feature_fusion != "fusion":
        detail_df = pd.DataFrame({
            "question_id": [m.get("questionId", m.get("sample_id", idx)) for idx, m in enumerate(sample_metas)],
            "question": questions,
            "enhanced_question": enhanced_prompts,
            "predicted_answer": preds,
            "ground_truth": refs,
            "score": scores,
        })
    else:
        detail_df = pd.DataFrame({
            "question_id": [m.get("questionId", m.get("sample_id", idx)) for idx, m in enumerate(sample_metas)],
            "question": questions,
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
        "use_feature_fusion": args.use_feature_fusion
    }
    json.dump(json_metrics, open(f"results/{basename}_metrics.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到 results/{basename}_*")


if __name__ == "__main__":
    main()