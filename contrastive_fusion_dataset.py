import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Any, Tuple
from loaders import VqaDataset
from tqdm import tqdm
import json
from synthetic_data_generator import SyntheticDataGenerator
import pyarrow.parquet as pq
import torch
import gc

class ContrastiveFusionDataset(VqaDataset):
    name = "contrastive_fusion"

    def __init__(self, split="train", sample_size=1000, data_dir="/root/autodl-tmp/dataset", **_):
        super().__init__(split)
        self.sample_size = sample_size
        self.split = split
        self.data_dir = data_dir
        self.synthetic_generator = SyntheticDataGenerator()
        
        # GPU内存优化设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32  # 小批量处理
        
        # 真正的流式加载：不缓存所有数据
        self._real_data_loaded = 0
        self._synthetic_data_loaded = 0
        self._real_data_target = sample_size // 2
        self._synthetic_data_target = sample_size - self._real_data_target
        
        # 预计算可用的真实数据量（使用高效方法）
        self._available_real_data = self._get_available_train_data_size()
        
        # 初始化数据源迭代器
        self._real_data_iterators = self._create_real_data_iterators()
        self._synthetic_data_iterator = self._create_synthetic_data_iterator()
    
    def _get_available_train_data_size(self):
        """获取可用的训练数据总量（高效版本，避免内存溢出）"""
        total_size = 0
        
        # 检查textVQA训练数据
        textvqa_dir = os.path.join(self.data_dir, "textVQA", "data")
        if os.path.exists(textvqa_dir):
            textvqa_files = [f for f in os.listdir(textvqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in textvqa_files:
                file_path = os.path.join(textvqa_dir, file)
                try:
                    # 使用pyarrow直接读取元数据，不加载整个文件
                    parquet_file = pq.ParquetFile(file_path)
                    total_size += parquet_file.metadata.num_rows
                except Exception as e:
                    print(f"读取textVQA文件{file}元数据失败: {e}")
        
        # 检查chartQA训练数据
        chartqa_dir = os.path.join(self.data_dir, "chartQA", "data")
        if os.path.exists(chartqa_dir):
            chartqa_files = [f for f in os.listdir(chartqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in chartqa_files:
                file_path = os.path.join(chartqa_dir, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    total_size += parquet_file.metadata.num_rows
                except Exception as e:
                    print(f"读取chartQA文件{file}元数据失败: {e}")
        
        # 检查docVQA训练数据
        docvqa_dir = os.path.join(self.data_dir, "docVQA", "data")
        if os.path.exists(docvqa_dir):
            docvqa_files = [f for f in os.listdir(docvqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in docvqa_files:
                file_path = os.path.join(docvqa_dir, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    total_size += parquet_file.metadata.num_rows
                except Exception as e:
                    print(f"读取docVQA文件{file}元数据失败: {e}")
        
        print(f"可用真实数据总量: {total_size}")
        return total_size
    
    def _create_real_data_iterators(self):
        """创建真实数据迭代器（流式读取，避免内存溢出）"""
        # textVQA数据迭代器
        textvqa_dir = os.path.join(self.data_dir, "textVQA", "data")
        if os.path.exists(textvqa_dir):
            textvqa_files = [f for f in os.listdir(textvqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in textvqa_files:
                file_path = os.path.join(textvqa_dir, file)
                try:
                    # 使用pyarrow的迭代器进行流式读取
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=self.batch_size):  # 小批量读取
                        df = batch.to_pandas()
                        for idx, row in df.iterrows():
                            yield self._convert_textvqa_sample(row, idx)
                        # 清理GPU内存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    print(f"读取textVQA文件{file}失败: {e}")
        
        # chartQA数据迭代器
        chartqa_dir = os.path.join(self.data_dir, "chartQA", "data")
        if os.path.exists(chartqa_dir):
            chartqa_files = [f for f in os.listdir(chartqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in chartqa_files:
                file_path = os.path.join(chartqa_dir, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                        df = batch.to_pandas()
                        for idx, row in df.iterrows():
                            yield self._convert_chartqa_sample(row, idx)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    print(f"读取chartQA文件{file}失败: {e}")
        
        # docVQA数据迭代器
        docvqa_dir = os.path.join(self.data_dir, "docVQA", "data")
        if os.path.exists(docvqa_dir):
            docvqa_files = [f for f in os.listdir(docvqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in docvqa_files:
                file_path = os.path.join(docvqa_dir, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                        df = batch.to_pandas()
                        for idx, row in df.iterrows():
                            yield self._convert_docvqa_sample(row, idx)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    print(f"读取docVQA文件{file}失败: {e}")
    
    def _create_synthetic_data_iterator(self):
        """创建合成数据迭代器"""
        idx = 0
        while True:
            yield self.synthetic_generator.generate_sample(idx)
            idx += 1
            # 定期清理内存
            if idx % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    
    def _get_next_textvqa_sample(self):
        """获取下一个textVQA样本"""
        try:
            # 从textVQA数据源获取样本
            for batch in self._create_textvqa_iterator():
                return batch
        except StopIteration:
            # 如果textVQA数据用完，使用合成数据
            return next(self._synthetic_data_iterator)
    
    def _get_next_chartqa_sample(self):
        """获取下一个chartvqa样本"""
        try:
            # 从chartvqa数据源获取样本
            for batch in self._create_chartqa_iterator():
                return batch
        except StopIteration:
            # 如果chartvqa数据用完，使用合成数据
            return next(self._synthetic_data_iterator)
    
    def _get_next_docvqa_sample(self):
        """获取下一个docvqa样本"""
        try:
            # 从docvqa数据源获取样本
            for batch in self._create_docvqa_iterator():
                return batch
        except StopIteration:
            # 如果docvqa数据用完，使用合成数据
            return next(self._synthetic_data_iterator)
    
    def _create_textvqa_iterator(self):
        """创建textVQA数据迭代器"""
        textvqa_dir = os.path.join(self.data_dir, "textVQA", "data")
        if os.path.exists(textvqa_dir):
            textvqa_files = [f for f in os.listdir(textvqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in textvqa_files:
                file_path = os.path.join(textvqa_dir, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                        df = batch.to_pandas()
                        for idx, row in df.iterrows():
                            yield self._convert_textvqa_sample(row, idx)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    print(f"读取textVQA文件{file}失败: {e}")
    
    def _create_chartqa_iterator(self):
        """创建chartvqa数据迭代器"""
        chartqa_dir = os.path.join(self.data_dir, "chartQA", "data")
        if os.path.exists(chartqa_dir):
            chartqa_files = [f for f in os.listdir(chartqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in chartqa_files:
                file_path = os.path.join(chartqa_dir, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                        df = batch.to_pandas()
                        for idx, row in df.iterrows():
                            yield self._convert_chartqa_sample(row, idx)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    print(f"读取chartQA文件{file}失败: {e}")
    
    def _create_docvqa_iterator(self):
        """创建docvqa数据迭代器"""
        docvqa_dir = os.path.join(self.data_dir, "docVQA", "data")
        if os.path.exists(docvqa_dir):
            docvqa_files = [f for f in os.listdir(docvqa_dir) 
                           if f.startswith("train") and f.endswith(".parquet")]
            for file in docvqa_files:
                file_path = os.path.join(docvqa_dir, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                        df = batch.to_pandas()
                        for idx, row in df.iterrows():
                            yield self._convert_docvqa_sample(row, idx)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    print(f"读取docVQA文件{file}失败: {e}")
    
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        """真正的流式加载：按需生成数据"""
        if idx >= self.sample_size:
            raise IndexError(f"Index {idx} out of range")
        
        # 前50%使用真实数据，不足部分用合成数据补充
        if idx < self._real_data_target:
            if self._real_data_loaded < self._available_real_data:
                try:
                    # 根据索引决定使用哪个数据源
                    # textVQA: 0-1/3, chartvqa: 1/3-2/3, docvqa: 2/3-1
                    source_idx = self._real_data_loaded % 3
                    
                    if source_idx == 0:  # textVQA
                        # 获取textVQA数据
                        sample = self._get_next_textvqa_sample()
                    elif source_idx == 1:  # chartvqa
                        # 获取chartvqa数据
                        sample = self._get_next_chartqa_sample()
                    else:  # docvqa
                        # 获取docvqa数据
                        sample = self._get_next_docvqa_sample()
                    
                    self._real_data_loaded += 1
                except StopIteration:
                    # 真实数据用完，使用合成数据
                    sample = next(self._synthetic_data_iterator)
                    self._synthetic_data_loaded += 1
            else:
                # 真实数据不足，使用合成数据
                sample = next(self._synthetic_data_iterator)
                self._synthetic_data_loaded += 1
        else:
            # 后50%使用合成数据
            sample = next(self._synthetic_data_iterator)
            self._synthetic_data_loaded += 1
        
        # 加载图像（优化内存使用）
        img = self._load_image(sample['image'])
        
        # 构建提示
        prompt = sample['question']
        
        # 获取答案
        answers = sample['answer']
        if not isinstance(answers, list):
            answers = [str(answers)]
        
        # 额外信息
        extra = {
            'questionId': idx,
            'source': sample['source'],
            'ocr_text': sample['ocr_text'],
            'dataset_type': 'contrastive_fusion',
            'split': self.split
        }
        
        return img, prompt, answers, extra
    
    def _convert_textvqa_sample(self, row, idx):
        """转换textVQA样本格式"""
        return {
            'image': row.get('image', {}),
            'question': row.get('question', f"TextVQA sample {idx}"),
            'answer': row.get('answers', [f"Answer {idx}"]),
            'source': 'textvqa',
            'ocr_text': row.get('ocr_text', f"OCR text {idx}")
        }
    
    def _convert_chartqa_sample(self, row, idx):
        """转换chartQA样本格式"""
        return {
            'image': row.get('image', {}),
            'question': row.get('query', f"ChartQA sample {idx}"),
            'answer': row.get('label', [f"Answer {idx}"]),
            'source': 'chartqa',
            'ocr_text': row.get('ocr_text', f"Chart data {idx}")
        }
    
    def _convert_docvqa_sample(self, row, idx):
        """转换docVQA样本格式"""
        return {
            'image': row.get('image', {}),
            'question': row.get('question', f"DocVQA sample {idx}"),
            'answer': row.get('answers', [f"Answer {idx}"]),
            'source': 'docvqa',
            'ocr_text': row.get('ocr_text', f"Document text {idx}")
        }
    
    def _load_image(self, image_field):
        """兼容 bytes / path / PIL.Image（优化内存使用）"""
        try:
            if isinstance(image_field, dict) and 'bytes' in image_field:
                return Image.open(io.BytesIO(image_field['bytes'])).convert('RGB')
            if isinstance(image_field, str):
                return Image.open(image_field).convert('RGB')
            if isinstance(image_field, Image.Image):
                return image_field.convert('RGB')
            raise ValueError(f"无法识别的image字段类型: {type(image_field)}")
        except Exception as e:
            # 如果图像加载失败，创建一个空白图像
            print(f"图像加载失败: {e}")
            return Image.new('RGB', (224, 224), color='white')
    
    def save_as_parquet(self, output_dir="/root/autodl-tmp/dataset/trainingVQA"):
        """将数据集保存为parquet文件（流式生成并保存，优化内存使用）"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用进度条
        with tqdm(total=self.sample_size, desc=f"生成{self.split}数据集") as pbar:
            data_list = []
            
            # 流式生成数据，分批处理避免内存溢出
            batch_size = 500  # 增大batch_size以提高性能
            for i in range(0, self.sample_size, batch_size):
                batch_end = min(i + batch_size, self.sample_size)
                batch_data = []
                
                for j in range(i, batch_end):
                    sample = self[j]  # 使用__getitem__方法获取样本
                    
                    # 转换为可保存的格式
                    data_sample = {
                        'image': sample[0],
                        'question': sample[1],
                        'answer': sample[2],
                        'source': sample[3]['source'],
                        'ocr_text': sample[3]['ocr_text'],
                        'questionId': j
                    }
                    batch_data.append(data_sample)
                    pbar.update(1)
                
                # 分批保存到临时文件
                temp_df = pd.DataFrame(batch_data)
                
                # 处理图像数据序列化
                if 'image' in temp_df.columns:
                    temp_df['image_bytes'] = temp_df['image'].apply(
                        lambda img: self._image_to_bytes(img)
                    )
                    # 删除原始的image列，避免Arrow无法序列化PIL对象
                    temp_df = temp_df.drop(columns=['image'])
                temp_file = os.path.join(output_dir, f"temp_{i}_{batch_end}.parquet")
                temp_df.to_parquet(temp_file, index=False)
                
                # 清理内存
                del temp_df, batch_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # 合并所有临时文件
            temp_files = [f for f in os.listdir(output_dir) if f.startswith("temp_") and f.endswith(".parquet")]
            if temp_files:
                dfs = []
                for temp_file in temp_files:
                    temp_path = os.path.join(output_dir, temp_file)
                    df = pd.read_parquet(temp_path)
                    dfs.append(df)
                    os.remove(temp_path)  # 删除临时文件
                
                final_df = pd.concat(dfs, ignore_index=True)
                # 修改文件名：训练集以train开头，验证集以val开头
                if self.split == "train":
                    output_file = os.path.join(output_dir, f"train_contrastive_fusion_{self.split}.parquet")
                elif self.split == "validation":
                    output_file = os.path.join(output_dir, f"val_contrastive_fusion_{self.split}.parquet")
                else:
                    output_file = os.path.join(output_dir, f"contrastive_fusion_{self.split}.parquet")
                
                final_df.to_parquet(output_file, index=False)
                print(f"数据集已保存到: {output_file}")
                
                return output_file
    
    def check_dataset_sampling(self, num_samples=10):
        """检查数据集随机采样是否正确"""
        print(f"\n=== 检查数据集采样情况 ===")
        print(f"数据集类型: {self.split}")
        print(f"总样本数: {self.sample_size}")
        print(f"真实数据目标数: {self._real_data_target}")
        print(f"可用真实数据数: {self._available_real_data}")
        
        # 随机检查几个样本
        import random
        random_indices = random.sample(range(self.sample_size), min(num_samples, self.sample_size))
        
        real_count = 0
        synthetic_count = 0
        
        for idx in random_indices:
            sample = self[idx]
            source = sample[3]['source']
            
            if source in ['textvqa', 'chartqa', 'docvqa']:
                real_count += 1
                print(f"样本 {idx}: 真实数据 ({source})")
            else:
                synthetic_count += 1
                print(f"样本 {idx}: 合成数据")
        
        print(f"\n采样统计:")
        print(f"真实数据样本数: {real_count}/{num_samples}")
        print(f"合成数据样本数: {synthetic_count}/{num_samples}")
        
        # 检查采样比例
        expected_real_ratio = min(0.5, self._available_real_data / self.sample_size)
        actual_real_ratio = real_count / num_samples
        
        print(f"期望真实数据比例: {expected_real_ratio:.2f}")
        print(f"实际真实数据比例: {actual_real_ratio:.2f}")
        
        if abs(actual_real_ratio - expected_real_ratio) < 0.1:
            print("✓ 采样比例正常")
        else:
            print("⚠ 采样比例可能有问题")
        
        return {
            'real_count': real_count,
            'synthetic_count': synthetic_count,
            'expected_ratio': expected_real_ratio,
            'actual_ratio': actual_real_ratio
        }
    
    def _image_to_bytes(self, img):
        """将PIL图像转换为bytes（优化内存使用）"""
        try:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True)
            return img_byte_arr.getvalue()
        except Exception as e:
            print(f"图像转换失败: {e}")
            # 返回空白图像
            blank_img = Image.new('RGB', (224, 224), color='white')
            blank_byte_arr = io.BytesIO()
            blank_img.save(blank_byte_arr, format='PNG')
            return blank_byte_arr.getvalue()
    
    @staticmethod
    def metrics(preds: List[str], refs: List[Any]) -> Dict[str, float]:
        """对比学习评估指标"""
        from sklearn.metrics import accuracy_score
        
        pred_labels = [str(p).strip().lower() for p in preds]
        ref_labels = [str(r[0]).strip().lower() for r in refs]
        
        accuracy = accuracy_score(ref_labels, pred_labels)
        
        return {
            'accuracy': float(accuracy),
            'total_samples': len(preds)
        }

# 数据集注册和测试
if __name__ == "__main__":
    # 创建训练集
    train_dataset = ContrastiveFusionDataset(split="train", sample_size=1000)
    print(f"训练集大小: {len(train_dataset)}")
    
    # 检查训练集采样情况
    train_dataset.check_dataset_sampling(num_samples=20)
    
    # 保存为parquet文件
    train_dataset.save_as_parquet()
    
    # 创建验证集
    val_dataset = ContrastiveFusionDataset(split="validation", sample_size=200)
    print(f"\n验证集大小: {len(val_dataset)}")
    
    # 检查验证集采样情况
    val_dataset.check_dataset_sampling(num_samples=10)
    
    # 保存为parquet文件
    val_dataset.save_as_parquet()
    
    # 查看样本示例
    print("\n=== 样本示例 ===")
    sample = train_dataset[0]
    print(f"图像类型: {type(sample[0])}")
    print(f"提示: {sample[1]}")
    print(f"答案: {sample[2]}")
    print(f"额外信息: {sample[3]}")
    
    # 检查文件是否生成正确
    print("\n=== 文件检查 ===")
    output_dir = "/root/autodl-tmp/dataset/trainingVQA"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"输出目录中的文件: {files}")
        
        # 检查训练集文件
        train_files = [f for f in files if f.startswith("train_")]
        if train_files:
            print(f"✓ 训练集文件已生成: {train_files}")
        else:
            print("⚠ 训练集文件未找到")
        
        # 检查验证集文件
        val_files = [f for f in files if f.startswith("val_")]
        if val_files:
            print(f"✓ 验证集文件已生成: {val_files}")
        else:
            print("⚠ 验证集文件未找到")