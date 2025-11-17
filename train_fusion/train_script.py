import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import json
import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from llava_ocr_model import LLaVAOCRModel
import time
import psutil
import subprocess
import glob
import re

class VQADataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_length=2048, file_pattern=None):
        # 支持单个路径或多个路径列表
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        dfs = []
        all_files = []
        
        # 遍历所有数据路径
        for data_path in data_paths:
            # 检查data_path是文件还是目录
            if os.path.isdir(data_path) and file_pattern:
                # 获取所有匹配的文件
                files = glob.glob(os.path.join(data_path, file_pattern))
                if files:
                    all_files.extend(files)
                    # 读取所有文件并合并
                    for file in files:
                        try:
                            df = pd.read_parquet(file)
                            # 添加源文件信息
                            df['source_file'] = os.path.basename(file)
                            dfs.append(df)
                        except Exception as e:
                            print(f"读取文件 {file} 时出错: {e}")
                else:
                    print(f"未找到匹配的文件: {os.path.join(data_path, file_pattern)}")
            elif os.path.isfile(data_path):
                # 单文件模式
                try:
                    df = pd.read_parquet(data_path)
                    df['source_file'] = os.path.basename(data_path)
                    dfs.append(df)
                    all_files.append(data_path)
                except Exception as e:
                    print(f"读取文件 {data_path} 时出错: {e}")
        
        if not dfs:
            raise ValueError(f"未成功加载任何数据文件")
        
        # 合并所有DataFrame
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"成功加载 {len(all_files)} 个文件，共 {len(self.data)} 条数据")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 确定正确的列名（不区分大小写）
        self.image_col = self._find_column('image', self.data.columns)
        self.question_col = self._find_column('question', self.data.columns)
        self.answer_col = self._find_column('answer', self.data.columns)
    
    def _find_column(self, keyword, columns):
        """使用正则表达式查找包含关键词的列名（不区分大小写）"""
        pattern = re.compile(f'.*{keyword}.*', re.IGNORECASE)
        for col in columns:
            if pattern.match(col):
                return col
        return keyword  # 如果没找到，返回原始关键词
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 获取数据项
        item = self.data.iloc[idx]
        
        # 处理图像
        try:
            # 尝试获取图像数据
            image_data = item.get(self.image_col)
            
            # 情况1: 直接是二进制数据
            if isinstance(image_data, bytes):
                # 如果是二进制数据
                from io import BytesIO
                image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # 情况2: 是字典，并且包含'bytes'字段
            elif isinstance(image_data, dict) and 'bytes' in image_data and isinstance(image_data['bytes'], bytes):
                # 从字典的bytes字段加载图像
                from io import BytesIO
                image = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
            
            # 情况3: 是字符串，尝试作为文件路径加载
            elif isinstance(image_data, str):
                # 如果是字符串，尝试作为文件路径加载
                if os.path.exists(image_data):
                    image = Image.open(image_data).convert('RGB')
                else:
                    # 尝试相对路径或其他可能的路径格式
                    # 检查当前目录和常见图像目录
                    possible_paths = [
                        image_data,
                        os.path.join('/root/autodl-tmp/dataset', image_data),
                        os.path.join('/root/autodl-tmp/dataset/finetuneVQA/data', image_data)
                    ]
                    found = False
                    for path in possible_paths:
                        if os.path.exists(path):
                            image = Image.open(path).convert('RGB')
                            found = True
                            break
                    if not found:
                        # 创建一个空白图像作为占位符
                        image = Image.new('RGB', (224, 224), color='white')
            else:
                # 如果是其他格式
                # 创建一个空白图像作为占位符
                image = Image.new('RGB', (224, 224), color='white')
        except Exception as e:
            # 静默处理异常，不打印错误信息
            image = Image.new('RGB', (224, 224), color='white')
        
        # 处理问题和答案
        try:
            question = str(item.get(self.question_col, ''))
            answer = str(item.get(self.answer_col, ''))
        except Exception as e:
            # 静默处理异常，不打印错误信息
            question = ''
            answer = ''
        
        # 构造提示
        prompt = f"<image>{question}\n{answer}"
        
        # 编码文本
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        
        # 准备标签（忽略question部分）
        # 首先找到answer开始的位置
        question_encoding = self.tokenizer(
            f"<image>{question}",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        question_len = question_encoding.input_ids.shape[1]
        
        labels = encoding.input_ids.clone()
        labels[:, :question_len] = -100  # 忽略question部分
        
        return {
            'image': image,
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze(),
            'labels': labels.squeeze(),
            'original_question': question
        }

class VQACollator:
    def __init__(self, image_processor, model_config):
        # 保存图像处理器和模型配置
        self.image_processor = image_processor
        self.model_config = model_config
        
    def __call__(self, batch):
        # 分离各个字段
        images = [item['image'] for item in batch]  # 保留原始PIL图像
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        original_questions = [item.get('original_question', '') for item in batch]  # 获取原始问题文本
        
        # 堆叠其他张量
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        
        return {
            'images': images,  # 直接返回原始PIL图像
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
            'original_questions': original_questions
        }

class Trainer:
    def __init__(self, config):
        global global_config
        global_config = config
        
        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        # 获取训练阶段，默认为1
        self.training_stage = config.get('training_stage', 1)
        print(f"当前训练阶段: {self.training_stage}")
        # 确保日志目录存在
        self.log_dir = self.config['training_config'].get('log_dir', '.')
        os.makedirs(self.log_dir, exist_ok=True)
        # 日志文件路径，区分训练阶段
        self.log_file = os.path.join(self.log_dir, f'train_log_stage_{self.training_stage}')
        # 初始化日志文件
        with open(self.log_file, 'w') as f:
            f.write('global_step,train_loss,val_loss,lr,gpu_usage,gpu_memory,eval_time,epoch_time\n')
        
    def initialize(self):
        """初始化训练器"""
        # 加载LLaVA模型和tokenizer
        print("正在加载LLaVA模型和tokenizer...")
        from models.llava.llava.model.builder import load_pretrained_model
        from models.llava.llava.mm_utils import get_model_name_from_path
        from transformers import BitsAndBytesConfig
        
        # 配置量化参数（如果需要）
        load_4bit = (self.config['training_config'].get('bits', 16) == 4)
        
        # 指定本地llava模型路径
        model_path = self.config['model_config'].get('llava_model_path', '/root/autodl-tmp/model/llava_hug')
        model_name = get_model_name_from_path(model_path)
        
        # 配置加载参数 - 强制使用FP32权重
        kwargs = {
            "device_map": "auto",
            "dtype": torch.float32
        }
        
        # 确保不使用低精度量化
        load_4bit = False
        
        # 即使配置要求4bit，也强制使用FP32以确保兼容性
        if self.config['training_config'].get('bits', 16) == 4:
            print("警告：4bit量化与FP32精度冲突，已强制使用FP32")
        
        # 加载模型，使用本地路径
        tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            **kwargs
        )
        
        # 重新设置分词器参数
        tokenizer.model_max_length = self.config['training_config']['max_length']
        tokenizer.padding_side = "right"
        
        # 加载OCR模型（可选）
        ocr_model = None
        try:
            from paddleocr import PaddleOCR
            print("正在加载OCR模型...")
            ocr_model = PaddleOCR(
                text_detection_model_dir=os.path.join(self.config['model_config']['ocr_model_path'], "det") \
                    if os.path.exists(os.path.join(self.config['model_config']['ocr_model_path'], "det")) else None,
                text_recognition_model_dir=os.path.join(self.config['model_config']['ocr_model_path'], "rec") \
                    if os.path.exists(os.path.join(self.config['model_config']['ocr_model_path'], "rec")) else None,
                textline_orientation_model_dir=None,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="ch",
                ocr_version="PP-OCRv5"
            )
        except Exception as e:
            print(f"OCR模型加载失败: {str(e)}")
        
        # 初始化LLaVAOCRModel，传入预加载的模型
        print("正在初始化LLaVAOCRModel...")
        self.model = LLaVAOCRModel(
            self.config,
            llava_model=llava_model,
            tokenizer=tokenizer,
            ocr_model=ocr_model
        )
        
        # 根据训练阶段设置模型模式
        if self.training_stage == 1:
            print("设置为阶段1: LLaVA预训练模式")
            # 调用模型的set_pretrain_mode方法冻结语言模型，只训练融合器和对齐模块
            self.model.set_pretrain_mode()
        elif self.training_stage == 2:
            print("设置为阶段2: 微调模式")
            # 调用模型的set_finetune_mode方法解冻所有参数
            self.model.set_finetune_mode()
            
            # 加载阶段1的检查点（new_params）
            stage1_checkpoint_path = self.config.get('stage1_checkpoint_path', 
                                                     '/root/autodl-tmp/Vqa_OCR/train_fusion/save/new_params.pth')
            if os.path.exists(stage1_checkpoint_path):
                print(f"正在加载阶段1的检查点: {stage1_checkpoint_path}")
                
                # 注释掉权重一致性检查逻辑
                # # 直接从阶段1检查点文件加载权重进行比较
                # print("\n===== 检查阶段1加载的三张量权重一致性 =====")
                # 
                # # 加载阶段1检查点文件中的权重
                # stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu')
                # 
                # # 打印检查点中的所有参数名称和值
                # print("\n阶段1检查点内容：")
                # print(f"检查点包含的键: {list(stage1_checkpoint.keys())}")
                # print(f"阶段1检查点中的参数数量: {len(stage1_checkpoint)}")
                # 
                # # 直接使用检查点作为权重字典（不再寻找'new_params'键）
                # stage1_weights = stage1_checkpoint
                
                # 加载阶段1的检查点到模型，并确保参数被移到正确设备
                self.model.load_new_params(stage1_checkpoint_path)
                # 将模型参数移到CUDA设备
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
               
                # all_weights_correct = True
                
                # # 检查每个关键参数是否与阶段1检查点中的权重相同
                # for param_name in ['ocr_text_projector', 'fusion_projector', 'fusion_lm_projector']:
                #     param_found = False
                #     for name, param in self.model.named_parameters():
                #         if param_name in name and param.requires_grad:
                #             param_found = True
                #             # 检查该参数是否在阶段1检查点中
                #             if name in stage1_weights:
                #                 # 获取检查点中的权重
                #                 checkpoint_weight = stage1_weights[name]
                                
                #                 # 计算当前模型中该参数的统计值
                #                 current_stats = (
                #                     param.data.mean().item(),
                #                     param.data.std().item(),
                #                     param.data.max().item(),
                #                     param.data.min().item()
                #                 )
                                
                #                 # 计算检查点中该参数的统计值
                #                 checkpoint_stats = (
                #                     checkpoint_weight.mean().item(),
                #                     checkpoint_weight.std().item(),
                #                     checkpoint_weight.max().item(),
                #                     checkpoint_weight.min().item()
                #                 )
                                
                #                 # 检查是否完全相同（允许极小的浮点误差）
                #                 is_identical = True
                #                 stats_diff = []
                                
                #                 for i in range(4):
                #                     diff = abs(current_stats[i] - checkpoint_stats[i])
                #                     stats_diff.append(diff)
                #                     if diff > 1e-6:
                #                         is_identical = False
                                
                #                 # 打印当前模型的权重统计值
                #                 print(f"当前模型权重统计 - {name}:")
                #                 print(f"    均值: {current_stats[0]:.8f}, 标准差: {current_stats[1]:.8f}")
                #                 print(f"    最大值: {current_stats[2]:.8f}, 最小值: {current_stats[3]:.8f}")
                                
                #                 if is_identical:
                #                     print(f"✓ {name} 权重与阶段1检查点完全一致")
                #                 else:
                #                     print(f"✗ 警告: {name} 权重与阶段1检查点不一致")
                #                     print(f"    差异详情: 均值差异={stats_diff[0]:.8f}, 标准差差异={stats_diff[1]:.8f}, 最大值差异={stats_diff[2]:.8f}, 最小值差异={stats_diff[3]:.8f}")
                #                     all_weights_correct = False
                #             else:
                #                 print(f"✗ 警告: {name} 在阶段1检查点中未找到")
                #                 all_weights_correct = False
                    
                #     if not param_found:
                #         print(f"✗ 警告: 未找到 {param_name} 参数")
                #         all_weights_correct = False
                
                # # 额外检查阶段1检查点中是否有未加载的关键参数
                # for weight_name in stage1_weights.keys():
                #     if any(param_name in weight_name for param_name in ['ocr_text_projector', 'fusion_projector', 'fusion_lm_projector']):
                #         param_exists = False
                #         for name, _ in self.model.named_parameters():
                #             if name == weight_name:
                #                 param_exists = True
                #                 break
                #         if not param_exists:
                #             print(f"✗ 警告: 阶段1检查点中的 {weight_name} 参数在当前模型中未找到")
                #             all_weights_correct = False
                
                # if all_weights_correct:
                #     print("✓ 所有三张量权重已成功从阶段1加载")
            else:
                print(f"警告: 未找到阶段1的检查点: {stage1_checkpoint_path}")
        
        # 启用梯度检查点
        if self.config['training_config'].get('gradient_checkpointing', False):
            self.model.llava_model.gradient_checkpointing_enable()
        
        # 强制使用混合精度训练
        self.use_amp = True
        # 初始化梯度缩放器
        self.scaler = GradScaler(device='cuda', 
                                 init_scale=2.**10,  # 默认是2.**16，降低初始缩放
                                 growth_interval=2000)  # 延长缩放因子调整间隔
        
        # 初始化数据加载器
        self._initialize_data_loaders()
        
        # 初始化优化器和调度器
        self._initialize_optimizer_and_scheduler()
        
        return self
        
    def _initialize_data_loaders(self):
        """初始化数据加载器"""
        # 根据训练阶段选择不同的数据集
        if self.training_stage == 1:
            # 阶段1：使用配置文件中的pretrain_data_config
            print("使用配置文件中的pretrain_data_config进行阶段1训练")
            # 从配置文件读取预训练数据配置
            train_data_paths = self.config['pretrain_data_config'].get('train_data_paths')
            val_data_paths = self.config['pretrain_data_config'].get('val_data_paths')
            train_file_pattern = self.config['pretrain_data_config'].get('train_file_pattern')
            val_file_pattern = self.config['pretrain_data_config'].get('val_file_pattern')
            num_workers = self.config['pretrain_data_config'].get('num_workers', 4)
        else:
            # 阶段2及以后：使用配置中的数据集
            print("使用配置中的数据集进行阶段2训练")
            # 优先使用新的多路径配置，如果不存在则使用旧的单路径配置
            train_data_paths = self.config['data_config'].get('train_data_paths')
            if not train_data_paths:
                # 向后兼容旧配置
                train_data_paths = [self.config['data_config'].get('train_data_path')]
            
            # 验证集路径
            val_data_paths = self.config['data_config'].get('val_data_paths')
            if not val_data_paths:
                # 向后兼容旧配置
                val_data_paths = [self.config['data_config'].get('val_data_path')]
            
            train_file_pattern = self.config['data_config'].get('train_file_pattern')
            val_file_pattern = self.config['data_config'].get('val_file_pattern')
            num_workers = self.config['data_config'].get('num_workers', 4)
        
        # 为训练集和验证集分别创建数据集，使用不同的文件模式
        train_dataset = VQADataset(
            train_data_paths,
            self.model.tokenizer,
            self.config['training_config']['max_length'],
            file_pattern=train_file_pattern
        )
        # 数据集切片调试，只使用前20个样本
        # train_dataset.data = train_dataset.data.iloc[:20]
        
        val_dataset = VQADataset(
            val_data_paths,
            self.model.tokenizer,
            self.config['training_config']['max_length'],
            file_pattern=val_file_pattern
        )
        # 数据集切片调试，只使用前20个样本
        # val_dataset.data = val_dataset.data.iloc[:20]
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training_config']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=VQACollator(self.model.image_processor, self.model.model_config)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training_config']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=VQACollator(self.model.image_processor, self.model.model_config)
        )
        
    def get_gpu_info(self):
        """获取GPU占用信息"""
        try:
            # 使用nvidia-smi命令获取GPU信息
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode('utf-8').strip()
            if result:
                gpu_util, gpu_mem = result.split(',')
                return float(gpu_util), float(gpu_mem)
            else:
                return 0.0, 0.0
        except:
            # 如果无法获取GPU信息，返回默认值
            return 0.0, 0.0
    
    def _initialize_optimizer_and_scheduler(self):
        """初始化优化器和学习率调度器"""
        # 获取可训练参数
        trainable_params = []
        lr = self.config['training_config']['learning_rate']
        projector_lr = self.config['training_config'].get('projector_learning_rate', lr * 5)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'mm_projector' in name or 'ocr_text_projector' in name:
                    # 为投影层使用较大的学习率
                    trainable_params.append({
                        'params': param,
                        'lr': projector_lr
                    })
                else:
                    trainable_params.append({
                        'params': param,
                        'lr': lr
                    })
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config['training_config']['weight_decay'],
            eps=1e-8
        )
        
        # 创建学习率调度器
        # 线性预热后余弦退火
        total_steps = len(self.train_loader) * self.config['training_config']['num_epochs'] // self.config['training_config']['gradient_accumulation_steps']
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=self.config['training_config']['warmup_steps']
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config['training_config']['warmup_steps'],
            eta_min=lr * 0.01
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config['training_config']['warmup_steps']]
        )
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        nan_count = 0  # 记录nan值出现的次数
        grad_nan_count = 0  # 记录梯度nan出现的次数
        global_step = 0  # 全局步数计数器
        # 保存四个层的梯度信息
        lora_grad_norm = 0.0
        ocr_text_projector_grad_norm = 0.0
        mm_projector_grad_norm = 0.0
        fusion_projector_grad_norm = 0.0
        # 记录epoch开始时间
        epoch_start_time = time.time()
        
        print(f"开始训练Epoch {epoch+1}/{self.config['training_config']['num_epochs']}")
        print(f"混合精度训练: {self.use_amp}")
        print(f"批量大小: {self.config['training_config']['batch_size']}")
        print(f"梯度累积步数: {self.config['training_config']['gradient_accumulation_steps']}")
        print("--- 开始梯度计算路径追踪 ---",)
        print("训练配置检查 - 梯度累积步数: {}, 混合精度: {}".format(self.config['training_config']['gradient_accumulation_steps'], self.use_amp))
        print("每个epoch结束时记录一次验证损失")
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training_config']['num_epochs']}")
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备，只对PyTorch张量调用to方法
            device = self.model.llava_model.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    
            # 前向传播 - 使用混合精度（FP16）
            with autocast(device_type='cuda'):
                outputs = self.model(
                    images=batch['images'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            
                # 计算损失
                loss = outputs.loss
            
            loss = loss / self.config['training_config']['gradient_accumulation_steps']
            
            # 记录当前批次的损失值
            # if step % 10 == 0 or step == len(self.train_loader) - 1:
            #     print(f"步骤 {step}, 批次损失: {loss.item() * self.config['training_config']['gradient_accumulation_steps']:.4f}, lr: {self.scheduler.get_last_lr()[0]:.8f}")
            
            # 反向传播与梯度缩放
            # 缩放损失并反向传播
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # 梯度累积
            if (step + 1) % self.config['training_config']['gradient_accumulation_steps'] == 0 or step == len(self.train_loader) - 1:
                # 注释掉权重打印逻辑
                # 记录权重更新前的张量值
                # if step == len(self.train_loader) - 1:  # 只打印最后一步
                #     print("\n===== 权重更新前的张量值 =====")
                #     # 获取一些关键参数的张量值
                #     for name, param in self.model.named_parameters():
                #         if param.requires_grad:
                #             # 重点关注以下几个模块的参数
                #             if ('lora' in name or 'mm_projector' in name or 'ocr_text_projector' in name or 
                #                 'fusion_projector' in name or 'fusion_lm_projector' in name):
                #                 # 只打印部分关键参数，避免输出过多
                #                 print(f"参数: {name}, 均值: {param.data.mean().item():.6f}, 标准差: {param.data.std().item():.6f}, "
                #                       f"最大值: {param.data.max().item():.6f}, 最小值: {param.data.min().item():.6f}")
                
                # 执行梯度裁剪（如果配置了梯度裁剪）
                gradient_clipping = self.config['training_config'].get('gradient_clipping', None)
                if gradient_clipping is not None:
                    # 解除梯度缩放以进行梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    # 执行梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
                    # print(f"步骤 {step}: 梯度范数 {grad_norm}")
                    
                    # 处理无效梯度
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"警告: 梯度范数无效 {grad_norm}")
                        # 手动将所有梯度设为小值
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad.zero_()
                        # 直接更新优化器（因为梯度已经被清零）
                        with self.scaler.no_grad():
                            self.optimizer.step()
                    else:
                        # 参数更新 - 使用FP32
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    # 重置梯度
                    self.optimizer.zero_grad()
                else:
                    # 没有梯度裁剪时，直接进行参数更新
                    
                    self.scaler.unscale_(self.optimizer)
                    
                    # 保存lora梯度范数，ocr_text_projector梯度范数，mm_projector梯度范数，fusion_projector和fusion_lm_projector梯度范数
                    lora_gradients = []
                    ocr_text_projector_gradients = []
                    mm_projector_gradients = []
                    fusion_projector_gradients = []
                    fusion_lm_projector_gradients = []

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if 'lora' in name:
                                lora_gradients.append(param.grad.data)
                            elif 'ocr_text_projector' in name:
                                ocr_text_projector_gradients.append(param.grad.data)
                            elif 'mm_projector' in name:
                                mm_projector_gradients.append(param.grad.data)
                            elif 'fusion_projector' in name and 'fusion_lm_projector' not in name:
                                fusion_projector_gradients.append(param.grad.data)
                            elif 'fusion_lm_projector' in name:
                                fusion_lm_projector_gradients.append(param.grad.data)

                    # 计算各部分梯度的范数
                    if lora_gradients:
                        lora_grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in lora_gradients]))
                    else:
                        lora_grad_norm = 0.0
                    
                    if ocr_text_projector_gradients:
                        ocr_text_projector_grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in ocr_text_projector_gradients]))
                    else:
                        ocr_text_projector_grad_norm = 0.0
                    
                    if mm_projector_gradients:
                        mm_projector_grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in mm_projector_gradients]))
                    else:
                        mm_projector_grad_norm = 0.0
                    
                    if fusion_projector_gradients:
                        fusion_projector_grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in fusion_projector_gradients]))
                    else:
                        fusion_projector_grad_norm = 0.0
                    
                    if fusion_lm_projector_gradients:
                        fusion_lm_projector_grad_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in fusion_lm_projector_gradients]))
                    else:
                        fusion_lm_projector_grad_norm = 0.0
                    # 执行优化器步骤更新权重
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # 注释掉权重打印逻辑
                    # 记录权重更新后的张量值
                    # if step == len(self.train_loader) - 1:  # 只打印最后一步
                    #     print("===== 权重更新后的张量值 =====")
                    #     # 获取相同参数更新后的张量值
                    #     for name, param in self.model.named_parameters():
                    #         if param.requires_grad:
                    #             # 与更新前保持相同的参数选择
                    #             if ('lora' in name or 'mm_projector' in name or 'ocr_text_projector' in name or 
                    #                 'fusion_projector' in name or 'fusion_lm_projector' in name):
                    #                 print(f"参数: {name}, 均值: {param.data.mean().item():.6f}, 标准差: {param.data.std().item():.6f}, "
                    #                       f"最大值: {param.data.max().item():.6f}, 最小值: {param.data.min().item():.6f}")
                    # 重置梯度
                    self.optimizer.zero_grad()
                    
                # 原有的梯度裁剪和优化器更新代码...
                # 学习率调度
                self.scheduler.step()
                global_step += 1
                
                self.optimizer.zero_grad()
            
            # 更新损失
            total_loss += loss.item() * self.config['training_config']['gradient_accumulation_steps']
            
            # 更新进度条
            base_postfix = {
                'loss': total_loss / (step + 1),
                'lr': self.scheduler.get_last_lr()[0],
            }
            grad_postfix = {}
            if 'lora_grad_norm' in locals():
                grad_postfix['lora_grad'] = lora_grad_norm
            if 'ocr_text_projector_grad_norm' in locals():
                grad_postfix['ocr_text_proj_grad'] = ocr_text_projector_grad_norm
            if 'mm_projector_grad_norm' in locals():
                grad_postfix['mm_projector_grad'] = mm_projector_grad_norm
            if 'fusion_projector_grad_norm' in locals():
                grad_postfix['fusion_proj_grad'] = fusion_projector_grad_norm
            if 'fusion_lm_projector_grad_norm' in locals():
                grad_postfix['fusion_lm_proj_grad'] = fusion_lm_projector_grad_norm
            
            # 分两行显示，先显示基础信息，再显示梯度信息
                progress_bar.set_postfix(base_postfix)
                # 打印基础信息行
                progress_bar.refresh()
                if grad_postfix:
                    print("\n", end="")  # 换行
                    # 打印梯度信息，不刷新进度条避免重复加载
                    print(" " * len(progress_bar.desc) + "  ", end="")  # 对齐进度条
                    for idx, (k, v) in enumerate(grad_postfix.items()):
                        if idx > 0:
                            print(", ", end="")
                        print(f"{k}: {v:.4f}", end="")
                    print()  # 换行
        
        # 确保训练完成统计一定会执行
        print(f"Epoch {epoch+1} 训练完成，平均损失: {total_loss / len(self.train_loader):.4f}")
        print(f"训练过程中检测到并处理了 {nan_count} 次nan/inf值")
        print(f"训练统计: 总步骤数={len(self.train_loader)}, 梯度nan/inf次数={grad_nan_count}, 总nan/inf次数={nan_count}")
        
        # 计算epoch训练时间
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} 训练时间: {epoch_time:.2f} 秒")
        
        return total_loss / len(self.train_loader), epoch_time
        
    def evaluate(self):
        """评估模型，处理OOM异常"""
        self.model.eval()
        total_loss = 0
        
        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Evaluating"):
                    # 移动到设备，只对PyTorch张量调用to方法
                    device = self.model.llava_model.device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    
                    # 前向传播 - 使用混合精度（FP16）
                    with autocast(device_type='cuda'):
                        outputs = self.model(
                            images=batch['images'],
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                    
                    # 累加损失
                    total_loss += outputs.loss.item()
            
            return total_loss / len(self.val_loader)
        except RuntimeError as e:
            # 检查是否是OOM错误
            if 'out of memory' in str(e).lower():
                print(f"警告: 验证阶段遇到CUDA OOM错误，跳过当前验证")
                # 释放缓存以尝试恢复内存
                torch.cuda.empty_cache()
                return 'oom'
            else:
                # 其他运行时错误重新抛出
                raise e
        
    def save_model(self, epoch, train_loss, val_loss):
        """保存模型，分别保存三类参数到不同文件"""
        # 创建保存根目录和epoch子目录，区分训练阶段
        base_save_dir = self.config['training_config']['save_dir']
        save_dir = os.path.join(base_save_dir, f"stage_{self.training_stage}")
        epoch_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 1. 保存LoRA权重参数和骨架（通过peft保存）
        if self.config['lora_config']['lora_enable']:
            # 使用PEFT库的save_pretrained方法保存LoRA权重到epoch目录
            lora_save_dir = os.path.join(epoch_dir, "peft_lora")
            self.model.llava_model.save_pretrained(lora_save_dir)
            print(f"LoRA权重已通过PEFT保存到 {lora_save_dir}")
                    
        # 2. 保存解冻参数（如mm_projector）
        if hasattr(self.model, 'unfrozen_params') and len(self.model.unfrozen_params) > 0:
            unfrozen_params_dict = {}
            # 先尝试直接通过get_model()方法获取mm_projector
            if hasattr(self.model, 'llava_model') and hasattr(self.model.llava_model, 'get_model'):
                try:
                    mm_projector = self.model.llava_model.get_model().mm_projector
                    for name, param in mm_projector.named_parameters():
                        full_name = f'llava_model.model.mm_projector.{name}'
                        unfrozen_params_dict[full_name] = param.to(torch.float32).clone()
                except Exception as e:
                    print(f"直接访问mm_projector失败: {str(e)}")
                    
            # 如果直接访问失败，尝试通过named_parameters查找
            if not unfrozen_params_dict:
                for name, param in self.model.named_parameters():
                    if 'mm_projector' in name:
                        unfrozen_params_dict[name] = param.to(torch.float32).clone()
            
            if unfrozen_params_dict:
                unfrozen_save_path = os.path.join(epoch_dir, "unfrozen_params.pth")
                unfrozen_save_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'unfrozen_params': unfrozen_params_dict
                }
                torch.save(unfrozen_save_dict, unfrozen_save_path)
                print(f"解冻参数已保存到 {unfrozen_save_path}，共 {len(unfrozen_params_dict)} 个参数")
        
        # 3. 保存新增参数（如ocr_text_projector和融合adapter）
        # 对于阶段1，使用模型的save_new_params方法保存融合器和对齐模块参数
        if self.training_stage == 1 and hasattr(self.model, 'save_new_params'):
            # 保存到特殊位置，方便阶段2加载
            stage1_checkpoint_path = os.path.join(base_save_dir, "new_params.pth")
            self.model.save_new_params(stage1_checkpoint_path)
        elif hasattr(self.model, 'new_params') and len(self.model.new_params) > 0:
            # 阶段2及以后，使用原来的保存逻辑
            new_params_dict = {}
            for param_name in self.model.new_params:
                # 尝试直接通过名称获取参数
                try:
                    # 构建参数路径访问链
                    parts = param_name.split('.')
                    module = self.model
                    for part in parts:
                        if hasattr(module, part):
                            module = getattr(module, part)
                        else:
                            break
                    else:
                        # 成功遍历所有部分，检查是否是参数
                        if isinstance(module, torch.nn.Parameter):
                            new_params_dict[param_name] = module.to(torch.float32).clone()
                except Exception as e:
                    # 如果通过名称路径访问失败，打印错误信息并尝试遍历所有参数
                    print(f"参数访问错误: {str(e)}")
                    for name, param in self.model.named_parameters():
                        if name == param_name:
                            new_params_dict[param_name] = param.to(torch.float32).clone()
                            break
            
            if new_params_dict:
                new_save_path = os.path.join(epoch_dir, "new_params.pth")
                new_save_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'new_params': new_params_dict
                }
                torch.save(new_save_dict, new_save_path)
                print(f"新增参数已保存到 {new_save_path}，共 {len(new_params_dict)} 个参数")
        
        # 4. 保存通用状态（优化器、调度器等）
        general_save_path = os.path.join(epoch_dir, "training_state.pth")
        general_save_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(general_save_dict, general_save_path)
        print(f"训练状态已保存到 {general_save_path}")
        
        # 5. 如果不使用LoRA，也保存完整模型权重作为备份
        if not self.config['lora_config']['lora_enable']:
            backup_save_path = os.path.join(epoch_dir, "full_model_backup.pth")
            backup_save_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_state_dict': {k: v.to(torch.float32) for k, v in self.model.state_dict().items()},
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }
            torch.save(backup_save_dict, backup_save_path)
            print(f"完整模型备份已保存到 {backup_save_path}")
        
        print(f"所有文件已保存到 {epoch_dir} 目录")
        
    def train(self):
        """执行训练"""
        best_val_loss = float('inf')
        
        # 记录训练开始时的初始张量值
        print("\n===== 训练开始时的初始张量值 =====")
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if ('lora' in name or 'mm_projector' in name or 'ocr_text_projector' in name or 
                    'fusion_projector' in name or 'fusion_lm_projector' in name):
                    print(f"参数: {name}, 均值: {param.data.mean().item():.6f}, 标准差: {param.data.std().item():.6f}, "
                          f"最大值: {param.data.max().item():.6f}, 最小值: {param.data.min().item():.6f}")
                    # 只打印前几个关键参数，避免输出过多
                    param_count += 1
                    if param_count > 5:
                        break
        
        for epoch in range(self.config['training_config']['num_epochs']):
            # 训练一个epoch
            train_loss, epoch_time = self.train_epoch(epoch)
            
            # 每个epoch结束后执行一次验证
            print(f"\n=== Epoch {epoch+1} 结束: 执行验证评估 ===")
            
            # 获取当前学习率
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 获取GPU信息
            gpu_util, gpu_mem = self.get_gpu_info()
            
            # 执行验证并记录时间
            start_time = time.time()
            val_loss = self.evaluate()
            eval_time = time.time() - start_time
            
            # 记录日志
            with open(self.log_file, 'a') as f:
                # 对于epoch结束的验证
                if val_loss == 'oom':
                    # OOM情况下记录特殊格式
                    f.write(f"epoch_{epoch+1},{train_loss:.6f},oom:null,{current_lr:.8f},{gpu_util:.2f},{gpu_mem:.2f},{eval_time:.4f},{epoch_time:.2f}\n")
                else:
                    f.write(f"epoch_{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.8f},{gpu_util:.2f},{gpu_mem:.2f},{eval_time:.4f},{epoch_time:.2f}\n")
            
            # 打印结果
            print(f"Epoch {epoch+1}/{self.config['training_config']['num_epochs']}")
            print(f"训练损失: {train_loss:.4f}")
            if val_loss == 'oom':
                print(f"验证损失: oom:null (因OOM跳过)")
                # OOM情况下仍然保存模型权重
                self.save_model(epoch, train_loss, val_loss)
            else:
                print(f"验证损失: {val_loss:.4f}")
                # 保存模型
                self.save_model(epoch, train_loss, val_loss)
                
                # 更新最佳验证损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"新的最佳验证损失: {best_val_loss:.4f}")
            
            print(f"日志已记录到: {self.log_file}")

if __name__ == "__main__":
    # 加载配置
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # 支持通过命令行参数指定训练阶段
    import sys
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        config['training_stage'] = int(sys.argv[1])
        print(f"通过命令行指定训练阶段: {config['training_stage']}")
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.initialize()
    trainer.train()