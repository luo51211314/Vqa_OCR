#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 VQA 批推理脚本（支持多种模型和指标）
支持插件式数据集（scienceqa / docvqa / gqa / chartqa）
支持多种模型（llava / qwen）
支持多种指标（anls / relaxed_accuracy / relaxed_accuracy_80）
支持专家模块增强和对比学习融合增强
"""

# ------------ 最优先设置环境变量 ------------
import os
# 设置Hugging Face镜像站点，避免从原始站点下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置Hugging Face缓存目录
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"
# 禁用Hugging Face的网络请求重试
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
import json
import argparse
import importlib
import subprocess
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from PIL import Image

# -------------- 统一数据集入口 --------------
from load_dataset import build_dataloader

# -------------- 专家模块 --------------
from choose_expert import ExpertChooser
from expert.expert_manager import ExpertManager

# -------------- 导入基础模型相关 --------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_fusion'))
from transformers import BitsAndBytesConfig

def get_gpu_memory_nvidia_smi():
    """
    使用 nvidia-smi 获取当前 GPU 显存使用情况
    Returns:
        tuple: (used_memory_mb, total_memory_mb, utilization_percent) 或 (None, None, None) 如果失败
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # 解析输出: "used, total, utilization"
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                used_mb = float(parts[0].strip())
                total_mb = float(parts[1].strip())
                util_percent = float(parts[2].strip()) if len(parts) > 2 else 0.0
                return used_mb, total_mb, util_percent
    except Exception as e:
        pass
    return None, None, None

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
    parser.add_argument("--dataset", choices=["scienceqa", "docvqa", "gqa", "chartqa", "testvqa", "tablevqa", "mydatavqa"], required=True)
    parser.add_argument("--split", default="validation", help="validation / test / val")
    parser.add_argument("--bs", type=int, default=4, help="batch size")
    parser.add_argument("--num_samples", type=int, default=None, help="仅调试：限制样本数")
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/llava_hug")
    parser.add_argument("--model_type", choices=["llava", "qwen", "blip", "mplug"], default="llava", help="模型类型")
    parser.add_argument("--metric_type", choices=["anls", "relaxed_accuracy", "relaxed_accuracy_80", "bleu", "weighted"], 
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
    # 时间测试参数
    parser.add_argument("--time", choices=["yes", "no"], default="no", 
                       help="是否测试推理时间: yes-测试, no-不测试（默认）")
    # OCR流水线消融实验参数
    parser.add_argument("--use_pipeline", action="store_true",
                       help="是否使用新的OCR流水线（DBSCAN聚类+拼写纠错+问题分类）")
    parser.add_argument("--pipeline_mode", type=str, default="full", 
                       choices=["full", "no_two_stage", "no_clustering", "no_spelling", "no_all", "no_stage"],
                       help="流水线模式: full-完整流水线(默认启用二阶段推理), no_two_stage-禁用二阶段推理(使用单阶段prompt), no_clustering-不使用聚类, no_spelling-不使用拼写纠错, no_all-不使用任何处理, no_stage-不使用阶段性prompt")
    parser.add_argument("--debug_pipeline", action="store_true",
                       help="是否输出流水线调试信息")
    parser.add_argument("--ocr_mode", type=str, default="normal", 
                        choices=["normal", "random", "LR-TB", "direct"],
                        help="OCR处理模式: normal(默认), random(随机打乱), LR-TB(列优先), direct(直接输出block)")
    # 保存路径参数
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/codes/Vqa_ocr/results",
                       help="结果保存目录（默认: /root/autodl-tmp/codes/Vqa_ocr/results）")
    
    args = parser.parse_args()

    # ---- 0. 模块初始化 ----
    active_experts = []
    expert_suffix = ""
    expert_manager = None
    fusion_suffix = ""
    expert_model = None  # 用于存储专家模型实例
    llava_model = None
    tokenizer = None
    
    # 预加载的BERT模型（用于OCR流水线）
    preloaded_models = {}
    
    # ---- 1. 数据集 ----
    loader = build_dataloader(args.dataset, args.split, batch_size=args.bs, num_workers=4)
    if args.num_samples:
        loader.dataset.df = loader.dataset.df[:args.num_samples]

    # ---- 2. 模型 ----
    device = torch.device("cuda")
    model = None
    
    # 动态导入model_loader，避免初始化时的导入错误
    from model_loader import get_model_loader
    
    # 统一从model_loader加载模型
    print(f"加载模型: {args.model_path}")
    model_loader = get_model_loader(args.model_type)
    tokenizer, model, processor, context_len = model_loader.load_model(args.model_path)
    image_processor = processor
    
    # 对于Qwen模型，我们不需要调用model.to，因为它已经用device_map加载了
    if args.model_type != "qwen":
        # 将模型设置为半精度模式以支持fp16权重
        model.to(device, dtype=torch.float16)
    print("模型加载成功")
    
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
                
                # 如果使用OCR专家且启用流水线，预加载BERT模型
                if expert_name == "ocr" and args.use_pipeline:
                    print("\n[预加载] 开始加载OCR流水线所需的BERT模型...")
                    try:
                        import sys
                        algorithm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Algorithm")
                        if algorithm_path not in sys.path:
                            sys.path.insert(0, algorithm_path)
                        
                        from transformers import AutoTokenizer, AutoModel
                        from transformers.utils.hub import cached_file
                        
                        def check_model_cached(model_name):
                            """检查模型是否已在本地缓存"""
                            try:
                                # 尝试获取配置文件路径，如果成功说明模型已缓存
                                cached_path = cached_file(model_name, "config.json", _raise_exceptions_for_missing_entries=False)
                                return cached_path is not None
                            except:
                                return False
                        
                        # 加载问题分类模型 (bert-tiny)
                        bert_model_name = "prajjwal1/bert-tiny"
                        print(f"[预加载] 加载问题分类模型: {bert_model_name}")
                        
                        # 检查模型是否已缓存
                        bert_cached = check_model_cached(bert_model_name)
                        if bert_cached:
                            print(f"[预加载] 模型已缓存，使用离线模式加载")
                            # 临时启用离线模式
                            os.environ["HF_HUB_OFFLINE"] = "1"
                        else:
                            print(f"[预加载] 模型未缓存，允许在线下载")
                            # 临时禁用离线模式以允许下载
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        
                        qa_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
                        qa_model = AutoModel.from_pretrained(bert_model_name)
                        qa_model.eval()
                        qa_model = qa_model.to(device)
                        preloaded_models["question_classifier"] = {
                            "tokenizer": qa_tokenizer,
                            "model": qa_model,
                            "device": device
                        }
                        print(f"[预加载] 问题分类模型加载成功")
                        
                        # 只有LLaVA模型才添加LLaVA相关预加载模型
                        if args.model_type != "qwen":
                            # 添加LLaVA模型到预加载模型中
                            preloaded_models["llava_model"] = model
                            preloaded_models["llava_tokenizer"] = tokenizer
                            preloaded_models["llava_processor"] = image_processor
                            preloaded_models["device"] = device
                        
                        # 恢复离线模式设置
                        os.environ["HF_HUB_OFFLINE"] = "1"
                        
                        print("[预加载] 所有BERT模型加载完成\n")
                        
                    except Exception as e:
                        print(f"[预加载警告] BERT模型加载失败: {e}")
                        print("[预加载警告] 将在需要时动态加载模型\n")
                        # 确保恢复离线模式
                        os.environ["HF_HUB_OFFLINE"] = "1"
                
                # 默认启用二阶段推理，除非指定了no_two_stage
                if args.use_pipeline and args.pipeline_mode != "no_two_stage":
                    print("\n[预加载] 启用二阶段推理模式，预加载VLM模型...")
                    preloaded_models["vlm_model"] = model
                    preloaded_models["vlm_processor"] = processor
                    print(f"[预加载] VLM模型已添加到预加载模型中")
                
                try:
                    expert_config = ExpertChooser.get_expert_config(expert_name)
                    expert_model = expert_manager.initialize_expert(expert_name, **expert_config)
                    
                    # 将预加载的模型传递给expert_manager
                    if preloaded_models:
                        expert_manager.set_preloaded_models(preloaded_models)
                    
                    expert_suffix = f"_{expert_name}"
                    print(f"专家模块初始化成功: {expert_name}")
                except Exception as e:
                    print(f"专家模块初始化失败: {e}")
                    import traceback
                    traceback.print_exc()
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
            
            # 只在需要时导入LLaVAOCRModel
            from train_fusion.llava_ocr_model import LLaVAOCRModel
            
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
    pipeline_stats_list = []  # 用于存储pipeline统计信息（如二阶段推理的OCR块）
    
    # 非fusion模式下才创建enhanced_prompts
    if args.use_feature_fusion != "fusion":
        enhanced_prompts = []
    
    # 时间统计
    inference_times = []
    prompt_lengths = []
    
    # 显存统计 - PyTorch API
    gpu_memory_stats = {
        "max_memory_allocated": [],
        "avg_memory_allocated": [],
        "max_memory_reserved": [],
        "avg_memory_reserved": []
    }
    # 显存统计 - nvidia-smi (实际GPU显存占用)
    nvidia_smi_memory = []
    
    start = time.time()

    for imgs, prompts, answers, extras in tqdm(loader, desc=f"{args.dataset}-{args.split}-{args.metric_type}"):
        batch_preds = []
        batch_processed_prompts = []
        batch_original_prompts = []
        batch_pipeline_stats = []  # 用于存储pipeline的统计信息（如二阶段推理的OCR块）
        
        # 非fusion模式下才创建batch_enhanced_prompts
        if args.use_feature_fusion != "fusion":
            batch_enhanced_prompts = []
        
        for i in range(len(imgs)):
            # 基础处理
            original_prompt = prompts[i]
            
            # 获取问题类型用于确定权重
            question_type = extras[i].get("question_type", "") if extras else ""
            
            # 根据问题类型确定权重
            if "Trend" in question_type:
                alpha, beta = 0.3, 0.7
            elif "Comparison" in question_type:
                alpha, beta = 0.4, 0.6
            elif "Comprehensive" in question_type or "Comprehension" in question_type:
                alpha, beta = 0.2, 0.8
            elif "Calculation" in question_type:
                alpha, beta = 0.7, 0.3
            elif "Location" in question_type:
                alpha, beta = 0.7, 0.3
            else:
                alpha, beta = 0.6, 0.4
            
            if args.use_feature_fusion == "fusion":
                # 使用LLaVAOCRModel的generate方法进行推理，通过autocast启用fp16推理
                sample_start = time.time() if args.time == "yes" else None
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
                if args.time == "yes":
                    sample_end = time.time()
                    inference_times.append(sample_end - sample_start)
                    prompt_lengths.append(len(original_prompt))
                    # 记录显存 - PyTorch API
                    if torch.cuda.is_available():
                        max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                        cur_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                        max_res = torch.cuda.max_memory_reserved() / 1024**3  # GB
                        cur_res = torch.cuda.memory_reserved() / 1024**3  # GB
                        gpu_memory_stats["max_memory_allocated"].append(max_mem)
                        gpu_memory_stats["avg_memory_allocated"].append(cur_mem)
                        gpu_memory_stats["max_memory_reserved"].append(max_res)
                        gpu_memory_stats["avg_memory_reserved"].append(cur_res)
                        torch.cuda.reset_peak_memory_stats()
                    # 记录显存 - nvidia-smi (实际GPU显存占用)
                    used_mb, total_mb, util_percent = get_gpu_memory_nvidia_smi()
                    if used_mb is not None:
                        nvidia_smi_memory.append(used_mb / 1024)  # 转换为GB
            else:
                if args.model_type == "qwen":
                    # Qwen2.5-VL模型推理
                    sample_start = time.time() if args.time == "yes" else None
                    
                    def qwen_inference(image, processed_prompt, max_new_tokens, max_size=1024):
                        """执行Qwen推理，支持OOM重试"""
                        # 调整图片大小
                        width, height = image.size
                        if max(width, height) > max_size:
                            ratio = max_size / max(width, height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            image = image.resize((new_width, new_height), Image.LANCZOS)
                        
                        # 构建对话
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": processed_prompt}
                                ]
                            }
                        ]
                        
                        text = processor.apply_chat_template(messages, add_generation_prompt=True)
                        inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
                        
                        with torch.no_grad():
                            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                        
                        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                        pred = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        return pred
                    
                    try:
                        image = imgs[i]
                        question = original_prompt
                        
                        # 专家模块处理
                        enhanced_prompt = original_prompt
                        pipeline_stats = {}  # 初始化pipeline_stats
                        if active_experts and expert_manager:
                            try:
                                question_type = extras[i].get("question_type", None) if extras else None
                                result = expert_manager.process_with_experts(
                                    imgs[i], original_prompt, active_experts,
                                    use_pipeline=args.use_pipeline,
                                    pipeline_mode=args.pipeline_mode,
                                    question_type=question_type,
                                    debug=args.debug_pipeline,
                                    ocr_mode=args.ocr_mode
                                )
                                # 处理返回值，可能是tuple(prompt, stats)或只有prompt
                                if isinstance(result, tuple):
                                    enhanced_prompt, pipeline_stats = result
                                else:
                                    enhanced_prompt = result
                            except Exception as e:
                                print(f"专家模块处理失败: {e}")
                        
                        # 使用model_loader处理prompt（添加提示词）
                        # 检查是否是二阶段推理模式（full_prompt已包含完整提示）
                        if "full_prompt" in pipeline_stats:
                            processed_prompt = enhanced_prompt
                        else:
                            processed_prompt = model_loader.process_prompt(enhanced_prompt, args.metric_type)
                        
                        # 尝试推理，OOM时自动重试（默认max_size=1024，减少OOM概率）
                        max_new_tokens = inference_config['max_new_tokens']
                        max_size = 1024
                        pred = None

                        for attempt in range(3):  # 最多重试3次
                            try:
                                pred = qwen_inference(image, processed_prompt, max_new_tokens, max_size)
                                break  # 成功则退出重试
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    print(f"OOM错误，尝试清理显存并减小图像尺寸重试 (尝试 {attempt + 1}/3)")
                                    torch.cuda.empty_cache()
                                    max_size = max_size // 2  # 减半图像尺寸: 1024->512->256
                                    max_new_tokens = max(64, max_new_tokens // 2)  # 减半token数
                                    if attempt == 2:  # 最后一次尝试
                                        raise e
                                else:
                                    raise e
                        
                        batch_preds.append(pred)
                        batch_processed_prompts.append(processed_prompt)
                        batch_enhanced_prompts.append(enhanced_prompt)
                        batch_original_prompts.append(original_prompt)
                        batch_pipeline_stats.append(pipeline_stats)
                    except Exception as e:
                        print(f"Qwen模型推理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        batch_preds.append("Error: Failed to generate")
                        batch_processed_prompts.append(processed_prompt if 'processed_prompt' in locals() else original_prompt)
                        batch_enhanced_prompts.append(enhanced_prompt if 'enhanced_prompt' in locals() else original_prompt)
                        batch_original_prompts.append(original_prompt)
                        batch_pipeline_stats.append({})
                    if args.time == "yes":
                        sample_end = time.time()
                        inference_times.append(sample_end - sample_start)
                        prompt_lengths.append(len(enhanced_prompt))
                        # 记录显存 - PyTorch API
                        if torch.cuda.is_available():
                            max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                            cur_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                            max_res = torch.cuda.max_memory_reserved() / 1024**3  # GB
                            cur_res = torch.cuda.memory_reserved() / 1024**3  # GB
                            gpu_memory_stats["max_memory_allocated"].append(max_mem)
                            gpu_memory_stats["avg_memory_allocated"].append(cur_mem)
                            gpu_memory_stats["max_memory_reserved"].append(max_res)
                            gpu_memory_stats["avg_memory_reserved"].append(cur_res)
                            torch.cuda.reset_peak_memory_stats()
                        # 记录显存 - nvidia-smi (实际GPU显存占用)
                        used_mb, total_mb, util_percent = get_gpu_memory_nvidia_smi()
                        if used_mb is not None:
                            nvidia_smi_memory.append(used_mb / 1024)  # 转换为GB
                elif args.model_type == "blip":
                    # BLIP/InstructBLIP 模型推理
                    sample_start = time.time() if args.time == "yes" else None
                    
                    try:
                        image = imgs[i]
                        
                        # 专家模块处理
                        enhanced_prompt = original_prompt
                        pipeline_stats = {}
                        if active_experts and expert_manager:
                            try:
                                question_type = extras[i].get("question_type", None) if extras else None
                                result = expert_manager.process_with_experts(
                                    imgs[i], original_prompt, active_experts,
                                    use_pipeline=args.use_pipeline,
                                    pipeline_mode=args.pipeline_mode,
                                    question_type=question_type,
                                    debug=args.debug_pipeline,
                                    ocr_mode=args.ocr_mode
                                )
                                if isinstance(result, tuple):
                                    enhanced_prompt, pipeline_stats = result
                                else:
                                    enhanced_prompt = result
                            except Exception as e:
                                print(f"专家模块处理失败: {e}")
                        
                        # 处理 prompt（添加 BLIP 格式）
                        if "full_prompt" in pipeline_stats:
                            processed_prompt = enhanced_prompt
                        else:
                            processed_prompt = model_loader.process_prompt(enhanced_prompt, args.metric_type)
                        
                        # 使用 BLIP 的 generate 方法
                        pred = model_loader.generate([image], [processed_prompt], inference_config)[0]

                        batch_preds.append(pred)
                        batch_processed_prompts.append(processed_prompt)
                        batch_enhanced_prompts.append(enhanced_prompt)
                        batch_original_prompts.append(original_prompt)
                        batch_pipeline_stats.append(pipeline_stats)
                        
                        if args.time == "yes":
                            sample_end = time.time()
                            inference_times.append(sample_end - sample_start)
                            prompt_lengths.append(len(processed_prompt))
                            # 记录显存 - PyTorch API
                            if torch.cuda.is_available():
                                max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                                cur_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                                max_res = torch.cuda.max_memory_reserved() / 1024**3  # GB
                                cur_res = torch.cuda.memory_reserved() / 1024**3  # GB
                                gpu_memory_stats["max_memory_allocated"].append(max_mem)
                                gpu_memory_stats["avg_memory_allocated"].append(cur_mem)
                                gpu_memory_stats["max_memory_reserved"].append(max_res)
                                gpu_memory_stats["avg_memory_reserved"].append(cur_res)
                                torch.cuda.reset_peak_memory_stats()
                            # 记录显存 - nvidia-smi (实际GPU显存占用)
                            used_mb, total_mb, util_percent = get_gpu_memory_nvidia_smi()
                            if used_mb is not None:
                                nvidia_smi_memory.append(used_mb / 1024)  # 转换为GB

                    except Exception as e:
                        print(f"BLIP模型推理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        batch_preds.append("Error: Failed to generate")
                        batch_processed_prompts.append(original_prompt)
                        batch_enhanced_prompts.append(original_prompt)
                        batch_original_prompts.append(original_prompt)
                        batch_pipeline_stats.append({})
                        if args.time == "yes":
                            inference_times.append(0)
                            prompt_lengths.append(len(original_prompt))
                            # 记录显存 - PyTorch API
                            if torch.cuda.is_available():
                                max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                                cur_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                                max_res = torch.cuda.max_memory_reserved() / 1024**3  # GB
                                cur_res = torch.cuda.memory_reserved() / 1024**3  # GB
                                gpu_memory_stats["max_memory_allocated"].append(max_mem)
                                gpu_memory_stats["avg_memory_allocated"].append(cur_mem)
                                gpu_memory_stats["max_memory_reserved"].append(max_res)
                                gpu_memory_stats["avg_memory_reserved"].append(cur_res)
                                torch.cuda.reset_peak_memory_stats()
                            # 记录显存 - nvidia-smi (实际GPU显存占用)
                            used_mb, total_mb, util_percent = get_gpu_memory_nvidia_smi()
                            if used_mb is not None:
                                nvidia_smi_memory.append(used_mb / 1024)  # 转换为GB
                elif args.model_type == "mplug":
                    # mPLUG-Owl3 模型推理
                    sample_start = time.time() if args.time == "yes" else None
                    
                    try:
                        image = imgs[i]
                        
                        # 专家模块处理
                        enhanced_prompt = original_prompt
                        pipeline_stats = {}
                        if active_experts and expert_manager:
                            try:
                                question_type = extras[i].get("question_type", None) if extras else None
                                result = expert_manager.process_with_experts(
                                    imgs[i], original_prompt, active_experts,
                                    use_pipeline=args.use_pipeline,
                                    pipeline_mode=args.pipeline_mode,
                                    question_type=question_type,
                                    debug=args.debug_pipeline,
                                    ocr_mode=args.ocr_mode
                                )
                                if isinstance(result, tuple):
                                    enhanced_prompt, pipeline_stats = result
                                else:
                                    enhanced_prompt = result
                            except Exception as e:
                                print(f"专家模块处理失败: {e}")
                        
                        # 处理 prompt
                        if "full_prompt" in pipeline_stats:
                            processed_prompt = enhanced_prompt
                        else:
                            processed_prompt = model_loader.process_prompt(enhanced_prompt, args.metric_type)
                        
                        # 使用 mPLUG 的 generate 方法
                        config = model_loader.get_inference_config(args.metric_type)
                        pred = model_loader.generate(image, processed_prompt, config["max_new_tokens"])

                        batch_preds.append(pred)
                        batch_processed_prompts.append(processed_prompt)
                        batch_enhanced_prompts.append(enhanced_prompt)
                        batch_original_prompts.append(original_prompt)
                        batch_pipeline_stats.append(pipeline_stats)
                        
                        if args.time == "yes":
                            sample_end = time.time()
                            inference_times.append(sample_end - sample_start)
                            prompt_lengths.append(len(processed_prompt))
                            # 记录显存 - PyTorch API
                            if torch.cuda.is_available():
                                max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                                cur_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                                max_res = torch.cuda.max_memory_reserved() / 1024**3  # GB
                                cur_res = torch.cuda.memory_reserved() / 1024**3  # GB
                                gpu_memory_stats["max_memory_allocated"].append(max_mem)
                                gpu_memory_stats["avg_memory_allocated"].append(cur_mem)
                                gpu_memory_stats["max_memory_reserved"].append(max_res)
                                gpu_memory_stats["avg_memory_reserved"].append(cur_res)
                                torch.cuda.reset_peak_memory_stats()
                            # 记录显存 - nvidia-smi (实际GPU显存占用)
                            used_mb, total_mb, util_percent = get_gpu_memory_nvidia_smi()
                            if used_mb is not None:
                                nvidia_smi_memory.append(used_mb / 1024)  # 转换为GB

                    except Exception as e:
                        print(f"mPLUG模型推理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        batch_preds.append("Error: Failed to generate")
                        batch_processed_prompts.append(original_prompt)
                        batch_enhanced_prompts.append(original_prompt)
                        batch_original_prompts.append(original_prompt)
                        batch_pipeline_stats.append({})
                        if args.time == "yes":
                            inference_times.append(0)
                            prompt_lengths.append(len(original_prompt))
                            # 记录显存 - PyTorch API
                            if torch.cuda.is_available():
                                max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                                cur_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                                max_res = torch.cuda.max_memory_reserved() / 1024**3  # GB
                                cur_res = torch.cuda.memory_reserved() / 1024**3  # GB
                                gpu_memory_stats["max_memory_allocated"].append(max_mem)
                                gpu_memory_stats["avg_memory_allocated"].append(cur_mem)
                                gpu_memory_stats["max_memory_reserved"].append(max_res)
                                gpu_memory_stats["avg_memory_reserved"].append(cur_res)
                                torch.cuda.reset_peak_memory_stats()
                            # 记录显存 - nvidia-smi (实际GPU显存占用)
                            used_mb, total_mb, util_percent = get_gpu_memory_nvidia_smi()
                            if used_mb is not None:
                                nvidia_smi_memory.append(used_mb / 1024)  # 转换为GB
                else:
                    # LLaVA模型推理
                    enhanced_prompt = original_prompt
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
                    pipeline_stats = {}  # 初始化pipeline_stats
                    if active_experts and expert_manager:
                        try:
                            question_type = extras[i].get("question_type", None) if extras else None
                            result = expert_manager.process_with_experts(
                                imgs[i], original_prompt, active_experts,
                                use_pipeline=args.use_pipeline,
                                pipeline_mode=args.pipeline_mode,
                                question_type=question_type,
                                debug=args.debug_pipeline,
                                ocr_mode=args.ocr_mode
                            )
                            # 处理返回值，可能是tuple(prompt, stats)或只有prompt
                            if isinstance(result, tuple):
                                enhanced_prompt, pipeline_stats = result
                            else:
                                enhanced_prompt = result
                        except Exception as e:
                            print(f"专家模块处理失败: {e}")
                    
                    # 检查是否是二阶段推理模式（full_prompt已包含完整提示）
                    if "full_prompt" in pipeline_stats:
                        prompt_in = enhanced_prompt
                    else:
                        prompt_in = model_loader.process_prompt(enhanced_prompt, args.metric_type)
                    batch_processed_prompts.append(prompt_in)
                    batch_enhanced_prompts.append(enhanced_prompt)
                    batch_original_prompts.append(original_prompt)
                    batch_pipeline_stats.append(pipeline_stats)
                    
                    # Tokenize和生成
                    input_ids = model_loader.tokenizer_image_token(prompt_in, tokenizer, None, return_tensors="pt")
                    input_ids = input_ids.unsqueeze(0).to(device)
                    
                    # 使用模型已设置的fp16模式进行推理
                    sample_start = time.time() if args.time == "yes" else None
                    output_ids = model_loader.generate(
                            input_ids, image_tensor, None, image_sizes, inference_config
                        )
                    
                    pred = model_loader.decode(output_ids, tokenizer)
                    batch_preds.append(pred)
                    if args.time == "yes":
                        sample_end = time.time()
                        inference_times.append(sample_end - sample_start)
                        prompt_lengths.append(len(prompt_in))
                        # 记录显存 - PyTorch API
                        if torch.cuda.is_available():
                            max_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                            cur_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                            max_res = torch.cuda.max_memory_reserved() / 1024**3  # GB
                            cur_res = torch.cuda.memory_reserved() / 1024**3  # GB
                            gpu_memory_stats["max_memory_allocated"].append(max_mem)
                            gpu_memory_stats["avg_memory_allocated"].append(cur_mem)
                            gpu_memory_stats["max_memory_reserved"].append(max_res)
                            gpu_memory_stats["avg_memory_reserved"].append(cur_res)
                            torch.cuda.reset_peak_memory_stats()
                        # 记录显存 - nvidia-smi (实际GPU显存占用)
                        used_mb, total_mb, util_percent = get_gpu_memory_nvidia_smi()
                        if used_mb is not None:
                            nvidia_smi_memory.append(used_mb / 1024)  # 转换为GB

        # 收集结果
        preds.extend(batch_preds)
        refs.extend(answers)
        sample_metas.extend(extras)
        questions.extend(batch_original_prompts)
        
        # 收集pipeline_stats用于CSV输出
        pipeline_stats_list.extend(batch_pipeline_stats)
        
        # 非fusion模式下才扩展enhanced_prompts
        if args.use_feature_fusion != "fusion":
            enhanced_prompts.extend(batch_processed_prompts)

    elapsed = time.time() - start

    # ---- 6. 指标计算和保存 ----
    module = importlib.import_module(f"loaders.{args.dataset}")
    
    # 对于weighted指标，需要传递extra_info参数
    if args.metric_type == "weighted":
        metrics = module.Dataset.metrics(preds, refs, extra_info=sample_metas, metric_type=args.metric_type, lambda1=0.6)
    else:
        metrics = module.Dataset.metrics(preds, refs, metric_type=args.metric_type)
    
    print("\n=== 评测结果 ===")
    print(f"{args.metric_type}: {metrics[args.metric_type]}")
    print(f"total_samples: {metrics['total_samples']}")
    
    # 时间统计
    if args.time == "yes" and inference_times:
        total_inference_time = sum(inference_times)
        avg_inference_time = total_inference_time / len(inference_times)
        avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        print(f"\n=== 时间统计 ===")
        print(f"总推理时间: {total_inference_time:.2f}秒")
        print(f"平均单样本推理时间: {avg_inference_time:.2f}秒")
        print(f"平均prompt长度: {avg_prompt_length:.0f}字符")
        # 显存统计
        if gpu_memory_stats["max_memory_allocated"]:
            print(f"\n=== 显存统计 (PyTorch API) ===")
            print(f"最大显存分配: {max(gpu_memory_stats['max_memory_allocated']):.2f} GB")
            print(f"平均显存分配: {sum(gpu_memory_stats['avg_memory_allocated']) / len(gpu_memory_stats['avg_memory_allocated']):.2f} GB")
            print(f"最大显存预留: {max(gpu_memory_stats['max_memory_reserved']):.2f} GB")
            print(f"平均显存预留: {sum(gpu_memory_stats['avg_memory_reserved']) / len(gpu_memory_stats['avg_memory_reserved']):.2f} GB")
            # nvidia-smi 显存统计
            if nvidia_smi_memory:
                print(f"\n=== 显存统计 (nvidia-smi 实际占用) ===")
                print(f"最大显存占用: {max(nvidia_smi_memory):.2f} GB")
                print(f"平均显存占用: {sum(nvidia_smi_memory) / len(nvidia_smi_memory):.2f} GB")
    
    # 保存结果
    suffix = expert_suffix if expert_suffix else fusion_suffix

    # 添加 time 和 pipeline 后缀
    time_suffix = "_time" if args.time == "yes" else ""
    pipeline_suffix = ""
    if args.use_pipeline:
        pipeline_suffix = f"_pipeline_{args.pipeline_mode}"

    # 添加 ocr_mode 后缀
    ocr_mode_suffix = f"_{args.ocr_mode}" if args.ocr_mode != "normal" else ""

    basename = f"{args.dataset}_{args.split}_{args.model_type}_{args.metric_type}{suffix}{time_suffix}{pipeline_suffix}{ocr_mode_suffix}"
    
    # 如果使用OCR流水线，保存到指定的输出目录
    if args.use_pipeline:
        # 如果用户指定了output_dir，使用用户指定的目录，否则使用默认的ocr_pipeline_results
        if args.output_dir != "/root/autodl-tmp/codes/Vqa_ocr/results":
            output_dir = args.output_dir
        else:
            output_dir = "/root/autodl-tmp/codes/Vqa_ocr/ocr_pipeline_results"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    import pandas as pd
    score_key = f"{args.metric_type}_scores"
    scores = metrics.get(score_key, [0.0] * len(preds))
    
    # 当使用weighted指标时，添加原始未加权分数
    extra_columns = {}
    if args.metric_type == "weighted":
        # 添加relaxed_acc_keywords和BLEU的原始分数列
        extra_columns["relaxed_acc_keywords_score"] = metrics.get("relaxed_acc_keywords_scores", [0.0] * len(preds))
        extra_columns["bleu_score"] = metrics.get("bleu_scores", [0.0] * len(preds))
    
    # 提取pipeline_stats中的OCR信息用于调试
    for i, stats in enumerate(pipeline_stats_list):
        # 过滤后的OCR blocks (阶段1输出)
        filtered_blocks = stats.get("stage1", {}).get("filtered_blocks_str", "")
        if filtered_blocks:
            print(f"[调试] 样本 {i} 过滤后的OCR blocks: {filtered_blocks}")
    
    # 打印enhanced_prompt用于调试
    for i, prompt in enumerate(enhanced_prompts):
        if i < 3:  # 只打印前3个样本的prompt
            print(f"[调试] 样本 {i} enhanced_prompt: {prompt[:300]}..." if len(prompt) > 300 else f"[调试] 样本 {i} enhanced_prompt: {prompt}")
    
    # 根据模式决定DataFrame的列
    if args.use_feature_fusion != "fusion":
        detail_df = pd.DataFrame({
            "question_id": [m.get("questionId", m.get("sample_id", idx)) for idx, m in enumerate(sample_metas)],
            "question": questions,
            "enhanced_question": enhanced_prompts,
            "predicted_answer": preds,
            "ground_truth": refs,
            "answer_full": [m.get("answer_full", "") for m in sample_metas],
            "score": scores,
            **extra_columns
        })
    else:
        detail_df = pd.DataFrame({
            "question_id": [m.get("questionId", m.get("sample_id", idx)) for idx, m in enumerate(sample_metas)],
            "question": questions,
            "predicted_answer": preds,
            "ground_truth": refs,
            "answer_full": [m.get("answer_full", "") for m in sample_metas],
            "score": scores,
            **extra_columns
        })
    detail_df.to_csv(f"{output_dir}/{basename}_detail.csv", index=False)
    
    # 保存指标 - 针对MyDataVQA的特殊处理
    import json  # 确保json模块已被导入
    
    if args.dataset == "mydatavqa" and args.split == "test":
        # 按question_type分类计算指标
        question_types = [m.get("question_type", "") for m in sample_metas]
        
        # 定义需要分类的question_type及其权重
        type_weights = {
            "Trend": {"lambda1": 0.3, "lambda2": 0.7},
            "Comparison": {"lambda1": 0.4, "lambda2": 0.6},
            "Comprehensive Understanding": {"lambda1": 0.2, "lambda2": 0.8},
            "Numerical Calculation": {"lambda1": 0.7, "lambda2": 0.3},
            "Numerical_Calculation": {"lambda1": 0.7, "lambda2": 0.3},
            "Location": {"lambda1": 0.7, "lambda2": 0.3}
        }
        
        # 按分类筛选索引
        trend_indices = [i for i, qt in enumerate(question_types) if "Trend" in qt]
        comparison_indices = [i for i, qt in enumerate(question_types) if "Comparison" in qt]
        comprehension_indices = [i for i, qt in enumerate(question_types) if "Comprehensive" in qt or "Comprehension" in qt]
        calculation_indices = [i for i, qt in enumerate(question_types) if "Calculation" in qt]
        location_indices = [i for i, qt in enumerate(question_types) if "Location" in qt]
        
        # 计算各分类的指标
        def calculate_metrics_for_indices(indices, lambda1=0.6):
            if not indices:
                return None
            filtered_preds = [preds[i] for i in indices]
            filtered_refs = [refs[i] for i in indices]
            filtered_extra = [sample_metas[i] for i in indices]
            
            if args.metric_type == "weighted":
                filtered_metrics = module.Dataset.metrics(filtered_preds, filtered_refs, extra_info=filtered_extra, metric_type=args.metric_type, lambda1=lambda1)
            else:
                filtered_metrics = module.Dataset.metrics(filtered_preds, filtered_refs, metric_type=args.metric_type)
            return filtered_metrics
        
        # 为每种类型使用不同的权重计算
        trend_metrics = calculate_metrics_for_indices(trend_indices, lambda1=type_weights["Trend"]["lambda1"])
        comparison_metrics = calculate_metrics_for_indices(comparison_indices, lambda1=type_weights["Comparison"]["lambda1"])
        comprehension_metrics = calculate_metrics_for_indices(comprehension_indices, lambda1=type_weights["Comprehensive Understanding"]["lambda1"])
        calculation_metrics = calculate_metrics_for_indices(calculation_indices, lambda1=type_weights["Numerical Calculation"]["lambda1"])
        location_metrics = calculate_metrics_for_indices(location_indices, lambda1=type_weights["Location"]["lambda1"])
        
        # 构建JSON输出格式
        model_combination = "纯 LLaVA" if args.use_experts == "off" and args.use_feature_fusion == "off" else f"OCR+LLaVA"
        
        # 创建结果数组
        json_output = {
            "model_combination": model_combination,
            "results": []
        }
        
        # 添加Trend类型的结果
        if trend_metrics:
            json_output["results"].append({
                "question_type": "Trend",
                "weight": {
                    "kcs": type_weights["Trend"]["lambda1"],
                    "bleu": type_weights["Trend"]["lambda2"]
                },
                "score": {
                    "kcs": round(trend_metrics.get("relaxed_acc_keywords", 0.0), 4),
                    "bleu": round(trend_metrics.get("bleu", 0.0), 4)
                },
                "total_score": round(trend_metrics.get("weighted", 0.0), 4),
                "sample_count": len(trend_indices)
            })
        else:
            json_output["results"].append({
                "question_type": "Trend",
                "weight": {
                    "kcs": type_weights["Trend"]["lambda1"],
                    "bleu": type_weights["Trend"]["lambda2"]
                },
                "score": {
                    "kcs": 0.0,
                    "bleu": 0.0
                },
                "total_score": 0.0,
                "sample_count": 0
            })
        
        # 添加Comparison类型的结果
        if comparison_metrics:
            json_output["results"].append({
                "question_type": "Comparison",
                "weight": {
                    "kcs": type_weights["Comparison"]["lambda1"],
                    "bleu": type_weights["Comparison"]["lambda2"]
                },
                "score": {
                    "kcs": round(comparison_metrics.get("relaxed_acc_keywords", 0.0), 4),
                    "bleu": round(comparison_metrics.get("bleu", 0.0), 4)
                },
                "total_score": round(comparison_metrics.get("weighted", 0.0), 4),
                "sample_count": len(comparison_indices)
            })
        else:
            json_output["results"].append({
                "question_type": "Comparison",
                "weight": {
                    "kcs": type_weights["Comparison"]["lambda1"],
                    "bleu": type_weights["Comparison"]["lambda2"]
                },
                "score": {
                    "kcs": 0.0,
                    "bleu": 0.0
                },
                "total_score": 0.0,
                "sample_count": 0
            })
        
        # 添加Comprehension类型的结果
        if comprehension_metrics:
            json_output["results"].append({
                "question_type": "Comprehension",
                "weight": {
                    "kcs": type_weights["Comprehensive Understanding"]["lambda1"],
                    "bleu": type_weights["Comprehensive Understanding"]["lambda2"]
                },
                "score": {
                    "kcs": round(comprehension_metrics.get("relaxed_acc_keywords", 0.0), 4),
                    "bleu": round(comprehension_metrics.get("bleu", 0.0), 4)
                },
                "total_score": round(comprehension_metrics.get("weighted", 0.0), 4),
                "sample_count": len(comprehension_indices)
            })
        else:
            json_output["results"].append({
                "question_type": "Comprehension",
                "weight": {
                    "kcs": type_weights["Comprehensive Understanding"]["lambda1"],
                    "bleu": type_weights["Comprehensive Understanding"]["lambda2"]
                },
                "score": {
                    "kcs": 0.0,
                    "bleu": 0.0
                },
                "total_score": 0.0,
                "sample_count": 0
            })
        
        # 添加Calculation类型的结果
        if calculation_metrics:
            json_output["results"].append({
                "question_type": "Calculation",
                "weight": {
                    "kcs": type_weights["Numerical Calculation"]["lambda1"],
                    "bleu": type_weights["Numerical Calculation"]["lambda2"]
                },
                "score": {
                    "kcs": round(calculation_metrics.get("relaxed_acc_keywords", 0.0), 4),
                    "bleu": round(calculation_metrics.get("bleu", 0.0), 4)
                },
                "total_score": round(calculation_metrics.get("weighted", 0.0), 4),
                "sample_count": len(calculation_indices)
            })
        else:
            json_output["results"].append({
                "question_type": "Calculation",
                "weight": {
                    "kcs": type_weights["Numerical Calculation"]["lambda1"],
                    "bleu": type_weights["Numerical Calculation"]["lambda2"]
                },
                "score": {
                    "kcs": 0.0,
                    "bleu": 0.0
                },
                "total_score": 0.0,
                "sample_count": 0
            })
        
        # 添加Location类型的结果
        if location_metrics:
            json_output["results"].append({
                "question_type": "Location",
                "weight": {
                    "kcs": type_weights["Location"]["lambda1"],
                    "bleu": type_weights["Location"]["lambda2"]
                },
                "score": {
                    "kcs": round(location_metrics.get("relaxed_acc_keywords", 0.0), 4),
                    "bleu": round(location_metrics.get("bleu", 0.0), 4)
                },
                "total_score": round(location_metrics.get("weighted", 0.0), 4),
                "sample_count": len(location_indices)
            })
        else:
            json_output["results"].append({
                "question_type": "Location",
                "weight": {
                    "kcs": type_weights["Location"]["lambda1"],
                    "bleu": type_weights["Location"]["lambda2"]
                },
                "score": {
                    "kcs": 0.0,
                    "bleu": 0.0
                },
                "total_score": 0.0,
                "sample_count": 0
            })
        
        # 添加总体结果
        json_output["overall"] = {
            "score": {
                "kcs": round(metrics.get("relaxed_acc_keywords", 0.0), 4),
                "bleu": round(metrics.get("bleu", 0.0), 4)
            },
            "total_score": round(metrics.get("weighted", 0.0), 4),
            "total_samples": metrics.get("total_samples", 0)
        }
        
        # 添加时间统计信息（如果启用）
        if args.time == "yes" and inference_times:
            json_output["time_stats"] = {
                "total_time": round(total_inference_time, 2),
                "avg_time": round(avg_inference_time, 2),
                "avg_len_prompt": round(avg_prompt_length, 0)
            }
            # 添加显存统计信息
            if gpu_memory_stats["max_memory_allocated"]:
                json_output["memory_stats"] = {
                    "max_memory_allocated_gb": round(max(gpu_memory_stats["max_memory_allocated"]), 2),
                    "avg_memory_allocated_gb": round(sum(gpu_memory_stats["avg_memory_allocated"]) / len(gpu_memory_stats["avg_memory_allocated"]), 2),
                    "max_memory_reserved_gb": round(max(gpu_memory_stats["max_memory_reserved"]), 2),
                    "avg_memory_reserved_gb": round(sum(gpu_memory_stats["avg_memory_reserved"]) / len(gpu_memory_stats["avg_memory_reserved"]), 2)
                }
            # 添加 nvidia-smi 显存统计
            if nvidia_smi_memory:
                json_output["nvidia_smi_memory_stats"] = {
                    "max_memory_gb": round(max(nvidia_smi_memory), 2),
                    "avg_memory_gb": round(sum(nvidia_smi_memory) / len(nvidia_smi_memory), 2)
                }

        # 保存JSON
        json.dump(json_output, open(f"{output_dir}/{basename}_metrics.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)

        print(f"\n=== MyDataVQA 分类评测结果 ===")
        for item in json_output["results"]:
            print(f"{item['question_type']} 类型: KCS={item['score']['kcs']}, BLEU={item['score']['bleu']}, 加权得分={item['total_score']} (权重: KCS={item['weight']['kcs']}, BLEU={item['weight']['bleu']}, 样本数: {item['sample_count']})")
        print(f"总体: KCS={json_output['overall']['score']['kcs']}, BLEU={json_output['overall']['score']['bleu']}, 加权得分={json_output['overall']['total_score']} (总样本数: {json_output['overall']['total_samples']})")
    else:
        # 非MyDataVQA数据集，使用原有格式
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
        
        # 当使用weighted指标时，保存原始的未加权分数
        if args.metric_type == "weighted" and "relaxed_acc_keywords" in metrics and "bleu" in metrics:
            json_metrics["relaxed_acc_keywords"] = metrics["relaxed_acc_keywords"]
            json_metrics["bleu"] = metrics["bleu"]
            json_metrics["lambda1"] = metrics["lambda1"]
            json_metrics["lambda2"] = metrics["lambda2"]
        
        # 添加时间统计信息（如果启用）
        if args.time == "yes" and inference_times:
            json_metrics["time_stats"] = {
                "total_time": round(total_inference_time, 2),
                "avg_time": round(avg_inference_time, 2),
                "avg_len_prompt": round(avg_prompt_length, 0)
            }
            # 添加显存统计信息
            if gpu_memory_stats["max_memory_allocated"]:
                json_metrics["memory_stats"] = {
                    "max_memory_allocated_gb": round(max(gpu_memory_stats["max_memory_allocated"]), 2),
                    "avg_memory_allocated_gb": round(sum(gpu_memory_stats["avg_memory_allocated"]) / len(gpu_memory_stats["avg_memory_allocated"]), 2),
                    "max_memory_reserved_gb": round(max(gpu_memory_stats["max_memory_reserved"]), 2),
                    "avg_memory_reserved_gb": round(sum(gpu_memory_stats["avg_memory_reserved"]) / len(gpu_memory_stats["avg_memory_reserved"]), 2)
                }
            # 添加 nvidia-smi 显存统计
            if nvidia_smi_memory:
                json_metrics["nvidia_smi_memory_stats"] = {
                    "max_memory_gb": round(max(nvidia_smi_memory), 2),
                    "avg_memory_gb": round(sum(nvidia_smi_memory) / len(nvidia_smi_memory), 2)
                }

        # 保存JSON
        json.dump(json_metrics, open(f"{output_dir}/{basename}_metrics.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到 {output_dir}/{basename}_*")


if __name__ == "__main__":
    main()