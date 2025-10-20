# VQA_OCR 项目使用指南

## 1. 环境准备

### 创建并激活vqa_infer环境
```bash
# 创建conda环境
conda create -n vqa_infer python=3.10 -y

# 激活环境
conda activate vqa_infer

# 升级pip
pip install --upgrade pip
```

### 安装项目依赖
```bash
# 在vqa_infer环境下安装requirements_vqa_infer.txt中的所有依赖
pip install -r /root/autodl-tmp/codes/Vqa_ocr/requirements_vqa_infer.txt
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

# 下载PaddleOCR模型（用于OCR专家模块）
./hfd.sh PaddlePaddle/PP-OCRv5 --local-dir ppocr_hug
    # for CUDA11.8
    python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

    # for CUDA12.6
    python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

    # for CPU
    python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
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

## 3. 专家模块系统

### 专家模块架构
项目采用模块化的专家系统设计，每个专家模块负责处理特定类型的任务：
- **OCR专家**: 文本识别和提取
- **Chart专家**: 图表数据提取
- **Text专家**: 文本分析和处理

### 专家模块工作流程
1. 专家模块处理输入图像，提取结构化信息
2. 将处理结果转换为LLM可理解的提示词格式
3. 多个专家输出组合成最终的prompt输入给LLaVA模型

### 在expert_manager中增加专家
专家管理器会自动发现`expert/`目录下的专家模块文件（以`_expert.py`结尾）。要添加新专家：

1. 在`expert/`目录下创建新的专家模块文件，如`new_expert.py`
2. 继承`BaseExpert`基类，实现三个抽象方法：
   - `initialize()`: 初始化专家模型
   - `process()`: 处理输入图像
   - `to_prompt()`: 转换结果为prompt格式
3. 专家模块会自动注册到专家管理器中

### 专家模块上下游关系
专家模块的输出作为LLaVA模型的文本prompt输入，形成上下游处理链：


## 4. 数据路径配置

下载完成后，数据集会自动保存在以下路径：
- DocVQA: `/root/autodl-tmp/dataset/docVQA/data/`
- ChartQA: `/root/autodl-tmp/dataset/chartQA/data/`
- 模型文件: `/root/autodl-tmp/model/llava_hug/`
- PaddleOCR模型: `/root/autodl-tmp/model/ppocr_hug/`

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

## 5. 运行测试

使用run.sh脚本运行测试，支持专家模块配置：

### 基本用法
```bash
# 运行DocVQA测试，自动选择专家
bash run.sh docvqa test 4 "" llava llava anls auto

# 运行ChartQA测试，手动指定专家  
bash run.sh chartqa val 2 100 qwen qwen relaxed_accuracy manual:text,chart

# 禁用专家模块
bash run.sh docvqa test 4 50 llava llava anls off

# 只测试部分样本
bash run.sh docvqa test 4 50   # 只测试50条数据
```

### run.sh参数说明
- `dataset`: docvqa, chartqa, scienceqa
- `split`: test, val, train  
- `batch_size`: 批次大小
- `num_samples`: 样本数量（空字符串表示全部）
- `model_name`: llava, qwen（对应_hug文件夹名称）
- `model_type`: llava, qwen
- `metric_type`: anls, relaxed_accuracy, relaxed_accuracy_80
- `use_experts`: 专家模块使用模式
  - `auto`: 自动选择合适专家
  - `manual:text,chart`: 手动指定专家列表
  - `off`: 禁用专家模块

### 专家模块使用示例
```bash
# 使用OCR专家处理文档
bash run.sh docvqa test 4 "" llava llava anls manual:ocr

# 使用多个专家组合
bash run.sh chartqa val 2 100 qwen qwen relaxed_accuracy manual:ocr,chart

# 自动选择专家（根据数据集类型）
bash run.sh docvqa test 4 "" llava llava anls auto
```

## 6. 模型支持

项目支持以下模型（需要下载到对应的_hug文件夹）：
- LLaVA系列: `llava_hug`
- Qwen系列: `qwen_hug` 
- PaddleOCR系列: `ppocr_hug`
- 其他HuggingFace模型

模型会自动从 `/root/autodl-tmp/model/` 目录下查找对应的_hug文件夹。

## 7. 专家模块开发

### 创建新专家模块
在`expert/`目录下创建新文件`new_expert.py`:

```python
from .base_expert import BaseExpert
from typing import Dict, Any, Optional

class NewExpert(BaseExpert):
    """新专家模块示例"""
    
    def __init__(self):
        super().__init__("new")
    
    def initialize(self, model_path: Optional[str] = None, **kwargs):
        """初始化专家模型"""
        # 实现模型初始化逻辑
        self.initialized = True
    
    def process(self, image, question: Optional[str] = None) -> Dict[str, Any]:
        """处理输入图像"""
        # 实现图像处理逻辑
        return {"result": "processed_data"}
    
    def to_prompt(self, result: Dict[str, Any]) -> str:
        """转换为LLM提示词"""
        # 将处理结果转换为prompt格式
        return f"New Expert Output: {result['result']}"
```

专家模块会自动被发现并注册到专家管理器中。