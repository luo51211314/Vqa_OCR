import torch
from PIL import Image
import sys
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 获取目标项目路径（示例：项目B的根目录）
project_b_path = "/root/autodl-tmp/model/LLaVA"  # 替换为实际路径
# 添加到sys.path
sys.path.append(project_b_path)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# 设置模型路径和图像路径
model_path = "/root/autodl-tmp/models/llava_hug"  # 您的本地模型路径
image_file = "/root/autodl-tmp/code/Vqa_ocr/算法比较.png"  # 您的图像路径
question = "如图所示，哪个算法的执行效果更好, prim还是kruskal"  # 您的问题

# 获取模型名称（从路径中提取）
model_name = get_model_name_from_path(model_path)

# 加载预训练模型
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)

# 使用 eval_model 函数进行推理
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": model_name,
    "query": question,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

# 运行模型
result = eval_model(args)
print(result)