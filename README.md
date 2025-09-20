# VQA_OCR 项目使用指南

## 1. 环境准备


### 创建模型隔离环境并安装依赖（以llava为例）
```bash
cd ../models/llava
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## 2. 下载模型和数据集

### 下载hfd工具
```bash
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
```

### 安装aria2（如果未安装）
```bash
sudo apt update
sudo apt install aria2
```

### 设置HuggingFace镜像环境变量
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 如果下载模型报错，重新执行此命令
```

### 下载LLaVA模型到llava_hug文件夹
```bash
# 创建模型存储目录
mkdir -p /root/autodl-tmp/model
cd /root/autodl-tmp/model

# 下载LLaVA模型（会自动创建llava_hug文件夹）
./hfd.sh liuhaotian/llava-v1.5-7b --local-dir llava_hug

# 下载其他模型（如qwen）
./hfd.sh Qwen/Qwen2-VL-7B-Instruct --local-dir qwen_hug
```

### 下载数据集
```bash
# 创建数据集目录
mkdir -p /root/autodl-tmp/dataset

# 下载DocVQA数据集
cd /root/autodl-tmp/dataset
./hfd.sh tonyassi/docvqa --dataset --local-dir docVQA

# 下载ChartQA数据集
./hfd.sh ahmed-masry/chartqa --dataset --local-dir chartQA

# 下载ScienceQA数据集（示例）
./hfd.sh derek-thomas/ScienceQA --dataset --local-dir scienceQA
```

## 3. 数据路径配置

下载完成后，数据集会自动保存在以下路径：
- DocVQA: `/root/autodl-tmp/dataset/docVQA/data/`
- ChartQA: `/root/autodl-tmp/dataset/chartQA/data/`
- 模型文件: `/root/autodl-tmp/model/llava_hug/`

如果路径不同，需要修改对应的数据加载器文件：

### 修改DocVQA数据路径
编辑 `/root/autodl-tmp/codes/Vqa_ocr/loaders/docvqa.py`:
```python
self.data_dir = "/your/custom/path/to/docVQA/data"
```

### 修改ChartQA数据路径  
编辑 `/root/autodl-tmp/codes/Vqa_ocr/loaders/chartqa.py`:
```python
self.data_dir = "/your/custom/path/to/chartQA/data"
```

## 4. 自定义数据加载器

如果需要创建新的数据加载器，可以在 `loaders/` 目录下创建对应的Python文件：

### 创建新的数据加载器模板
```python
from loaders import VqaDataset
import pandas as pd
import os
from PIL import Image

class Dataset(VqaDataset):
    name = "your_dataset_name"

    def __init__(self, split="val", **_):
        super().__init__(split)
        self.data_dir = "/path/to/your/dataset/data"
        # 实现数据加载逻辑
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # 实现数据获取逻辑
        return img, prompt, answers, metadata
```

## 5. 运行测试

使用run.sh脚本运行测试：

### 基本用法
```bash
# 运行DocVQA测试
bash run.sh docvqa test 4 "" llava llava anls

# 运行ChartQA测试  
bash run.sh chartqa val 2 100 qwen qwen relaxed_accuracy

# 只测试部分样本
bash run.sh docvqa test 4 50   # 只测试50条数据
```

### 参数说明
- `dataset`: docvqa, chartqa, scienceqa
- `split`: test, val, train  
- `batch_size`: 批次大小
- `num_samples`: 样本数量（空字符串表示全部）
- `model_name`: llava, qwen（对应_hug文件夹名称）
- `model_type`: llava, qwen
- `metric_type`: anls, relaxed_accuracy, relaxed_accuracy_80

## 6. 模型支持

项目支持以下模型（需要下载到对应的_hug文件夹）：
- LLaVA系列: `llava_hug`
- Qwen系列: `qwen_hug` 
- 其他HuggingFace模型

模型会自动从 `/root/autodl-tmp/model/` 目录下查找对应的_hug文件夹。