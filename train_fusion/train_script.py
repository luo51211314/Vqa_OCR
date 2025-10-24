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

class VQADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048, file_pattern=None):
        # 检查data_path是文件还是目录
        if os.path.isdir(data_path) and file_pattern:
            # 获取所有匹配的文件
            files = glob.glob(os.path.join(data_path, file_pattern))
            if not files:
                raise ValueError(f"未找到匹配的文件: {os.path.join(data_path, file_pattern)}")
            # 读取所有文件并合并
            dfs = []
            for file in files:
                dfs.append(pd.read_parquet(file))
            self.data = pd.concat(dfs, ignore_index=True)
            print(f"成功加载 {len(files)} 个文件，共 {len(self.data)} 条数据")
        else:
            # 单文件模式
            self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 获取数据项
        item = self.data.iloc[idx]
        
        # 处理图像 - 使用image_bytes列（二进制图像数据）
        image_bytes = item['image_bytes']
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # 处理问题和答案
        question = item['question']
        answer = item['answer']
        
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
        # 确保日志目录存在
        self.log_dir = self.config['training_config'].get('log_dir', '.')
        os.makedirs(self.log_dir, exist_ok=True)
        # 日志文件路径
        self.log_file = os.path.join(self.log_dir, 'train_log')
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
        
        # 配置加载参数 - 使用FP32权重
        kwargs = {
            "device_map": "auto",
            "dtype": torch.float32
        }
        
        if load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        
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
        # 创建数据集
        train_dataset = VQADataset(
            self.config['data_config']['train_data_path'],
            self.model.tokenizer,
            self.config['training_config']['max_length'],
            file_pattern=self.config['data_config'].get('train_file_pattern')
        )
        # 数据集切片调试，只使用前20个样本
        train_dataset.data = train_dataset.data.iloc[:20]
        
        val_dataset = VQADataset(
            self.config['data_config']['val_data_path'],
            self.model.tokenizer,
            self.config['training_config']['max_length'],
            file_pattern=self.config['data_config'].get('val_file_pattern')
        )
        # 数据集切片调试，只使用前20个样本
        val_dataset.data = val_dataset.data.iloc[:20]
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training_config']['batch_size'],
            shuffle=True,
            num_workers=self.config['data_config']['num_workers'],
            collate_fn=VQACollator(self.model.image_processor, self.model.model_config)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training_config']['batch_size'],
            shuffle=False,
            num_workers=self.config['data_config']['num_workers'],
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
                # print(f"步骤 {step}: 开始梯度累积更新")
                
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
                    
                    # 保存lora梯度范数，ocr_text_projector梯度范数，mm_projector梯度范数和fusion_projector梯度范数
                    lora_gradients = []
                    ocr_text_projector_gradients = []
                    mm_projector_gradients = []
                    fusion_projector_gradients = []

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if 'lora' in name:
                                lora_gradients.append(param.grad.data)
                            elif 'ocr_text_projector' in name:
                                ocr_text_projector_gradients.append(param.grad.data)
                            elif 'mm_projector' in name:
                                mm_projector_gradients.append(param.grad.data)
                            elif 'fusion' in name and 'projector' in name:
                                fusion_projector_gradients.append(param.grad.data)

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
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
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
        # 创建保存根目录和epoch子目录
        save_dir = self.config['training_config']['save_dir']
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
        if hasattr(self.model, 'new_params') and len(self.model.new_params) > 0:
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
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.initialize()
    trainer.train()