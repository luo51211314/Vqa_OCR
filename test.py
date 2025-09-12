# import torch
# import sys
# import os
# import pandas as pd
# from PIL import Image
# import io
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score
# import json
# import tempfile
# import re

# project_b_path = "/root/autodl-tmp/models/LLaVA"
# sys.path.append(project_b_path)

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model

# # 环境设置
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式

# # 模型加载配置
# model_path = "/root/autodl-tmp/models/llava_hug"
# model_name = get_model_name_from_path(model_path)
# print(f"加载模型: {model_name}")

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=model_name
# )

# # 加载ScienceQA测试集
# data_path = "/root/autodl-tmp/datasets/scienceqa/data/test-00000-of-00001-f0e719df791966ff.parquet"
# test_data = pd.read_parquet(data_path)

# print(f"数据集大小: {len(test_data)} 条样本")
# print(f"有图像的样本: {test_data['image'].notna().sum()} 条")

# # 筛选有图像的样本
# image_data = test_data[test_data['image'].notna()].copy()
# print(f"将处理 {len(image_data)} 个有图像的样本")

# # 定义评估指标
# def calculate_metrics(predictions, references):
#     if not predictions or not references:
#         return {'accuracy': 0, 'f1_score': 0}
    
#     try:
#         # 将预测答案转换为数字格式
#         pred_numeric = []
#         for pred in predictions:
#             if isinstance(pred, str) and pred.strip().isdigit():
#                 pred_numeric.append(int(pred.strip()))
#             elif isinstance(pred, (int, float)):
#                 pred_numeric.append(int(pred))
#             else:
#                 # 对于文本答案，尝试提取第一个数字或设为默认值
#                 pred_numeric.append(-1)
        
#         ref_numeric = [int(ref) for ref in references]
        
#         acc = accuracy_score(ref_numeric, pred_numeric)
#         f1 = f1_score(ref_numeric, pred_numeric, average='macro')
        
#         return {
#             'accuracy': round(acc, 4),
#             'f1_score': round(f1, 4)
#         }
#     except Exception as e:
#         print(f"指标计算错误: {e}")
#         return {'accuracy': 0, 'f1_score': 0}

# # 改进的样本处理函数
# def process_sample(row, sample_idx):
#     try:
#         # 检查图像数据
#         if pd.isna(row['image']) or row['image'] is None:
#             print(f"样本 {sample_idx}: 无图像数据，跳过")
#             return None
        
#         # 创建临时文件保存图像
#         if isinstance(row['image'], dict) and 'bytes' in row['image']:
#             image_bytes = row['image']['bytes']
#         else:
#             print(f"样本 {sample_idx}: 图像格式不支持")
#             return None
        
#         # 创建临时文件
#         with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
#             tmp_file.write(image_bytes)
#             tmp_file_path = tmp_file.name
        
#         # 使用您提供的图像加载逻辑
#         try:
#             # 构建问题提示
#             question_text = row['question']
            
#             # 处理choices - 修复空列表问题
#             if isinstance(row['choices'], np.ndarray) and row['choices'].size > 0:
#                 # 将NumPy数组转换为Python列表
#                 choices = row['choices'].tolist()
#                 choices = [f"{i}. {str(choice)}" for i, choice in enumerate(choices) if choice is not None]
            
#             # 确保choices不为空
#             if not choices:
#                 choices = ['True', 'False']
            
#             question_prompt = f"{question_text}\n选项: {', '.join(map(str, choices))}, 请问选哪个选项，请输出选项前的数字"
#             print(question_prompt)
            
#             # 模型推理
#             args = type('Args', (), {
#                 "model_path": model_path,
#                 "model_base": None,
#                 "model_name": model_name,
#                 "query": question_prompt,
#                 "conv_mode": None,
#                 "image_file": tmp_file_path,
#                 "sep": ",",
#                 "temperature": 0.1,  # 稍微增加温度以获得更好的结果
#                 "top_p": 0.9,
#                 "num_beams": 2,
#                 "max_new_tokens": 128
#             })()
            
#             # 获取模型输出
#             output = eval_model(args)
            
#             # 清理临时文件
#             os.unlink(tmp_file_path)
            
#             return {
#                 'sample_id': sample_idx,
#                 'question': question_text,
#                 'choices': choices,
#                 'predicted_answer': output,
#                 'correct_answer': row['answer']
#             }
            
#         except Exception as e:
#             # 确保临时文件被清理
#             if os.path.exists(tmp_file_path):
#                 os.unlink(tmp_file_path)
#             raise e
            
#     except Exception as e:
#         print(f"处理样本 {sample_idx} 时出错: {str(e)}")
#         return None

# # 评估有图像的样本
# print("\n=== 开始样本推理 ===")
# sample_results = []
# all_predictions = []
# all_references = []

# # 只处理前10个有图像的样本进行测试
# test_samples = image_data.head(10)

# for idx, (_, row) in enumerate(test_samples.iterrows()):
#     result = process_sample(row, idx)
#     if result:
#         sample_results.append(result)
#         all_predictions.append(result['predicted_answer'])
#         all_references.append(result['correct_answer'])
        
#         print(f"\n样本 {idx + 1}:")
#         print(f"问题: {result['question']}")
#         print(f"选项: {result['choices']}")
#         print(f"模型答案: {result['predicted_answer']}")
#         print(f"正确答案: {result['correct_answer']}")
#     else:
#         print(f"\n样本 {idx + 1}: 处理失败")

# # 计算指标
# if sample_results:
#     metrics = calculate_metrics(all_predictions, all_references)
#     print("\n=== 评估结果 ===")
#     print(f"成功处理样本数: {len(sample_results)}")
#     print(f"准确率: {metrics['accuracy']}")
#     print(f"F1分数: {metrics['f1_score']}")
    
#     # 保存结果到CSV文件
#     output_df = pd.DataFrame({
#         'sample_id': [r['sample_id'] for r in sample_results],
#         'question': [r['question'] for r in sample_results],
#         'choices': [r['choices'] for r in sample_results],
#         'predicted_answer': [r['predicted_answer'] for r in sample_results],
#         'correct_answer': [r['correct_answer'] for r in sample_results]
#     })
    
#     output_path = '/root/autodl-tmp/code/Vqa_ocr/scienceqa_eval_results.csv'
#     output_df.to_csv(output_path, index=False)
#     print(f"\n结果已保存至: {output_path}")
    
#     # 打印样本示例
#     print("\n=== 样本推理示例 ===")
#     for i, result in enumerate(sample_results[:5]):
#         print(f"\n示例 {i + 1}:")
#         print(f"问题: {result['question']}")
#         print(f"选项: {result['choices']}")
#         print(f"模型答案: {result['predicted_answer']}")
#         print(f"正确答案: {result['correct_answer']}")
# else:
#     print("\n没有成功处理的样本")

# print("\n=== 处理完成 ===")

import torch
import sys
import os
import pandas as pd
from PIL import Image
import io
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import tempfile
import re
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------- 把 LLaVA 路径加入 sys.path -----------
project_b_path = "/root/autodl-tmp/models/LLaVA"
sys.path.append(project_b_path)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# ----------- 环境 & 模型 -----------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "/root/autodl-tmp/models/llava_hug"
model_name = get_model_name_from_path(model_path)
print(f"加载模型: {model_name}")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device_map=None,  # 强制禁用自动映射
    device="cuda",    # 强制加载到 cuda:0
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ----------- 数据集加载 -----------
data_path = "/root/autodl-tmp/datasets/scienceqa/data/test-00000-of-00001-f0e719df791966ff.parquet"
test_data = pd.read_parquet(data_path)
print(f"数据集大小: {len(test_data)} 条")
print(f"有图像的样本: {test_data['image'].notna().sum()} 条")
image_data = test_data[test_data['image'].notna()].reset_index(drop=True)
print(f"将处理 {len(image_data)} 个有图像的样本")

# ----------- 指标函数 -----------
def calculate_metrics(predictions, references):
    if not predictions or not references:
        return {"accuracy": 0, "f1_score": 0}
    try:
        pred_numeric = []
        for pred in predictions:
            s = str(pred).strip()
            if s.isdigit():
                pred_numeric.append(int(s))
            else:
                m = re.search(r"\d", s)
                pred_numeric.append(int(m.group()) if m else -1)
        ref_numeric = [int(r) for r in references]
        acc = accuracy_score(ref_numeric, pred_numeric)
        f1 = f1_score(ref_numeric, pred_numeric, average="macro")
        return {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}
    except Exception as e:
        print(f"指标计算错误: {e}")
        return {"accuracy": 0, "f1_score": 0}

# ----------- Dataset -----------
class SqaDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 图像
        if isinstance(row["image"], dict) and "bytes" in row["image"]:
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), color="white")
        # 文字
        choices = []
        if isinstance(row["choices"], np.ndarray) and row["choices"].size > 0:
            choices = [f"{i}. {c}" for i, c in enumerate(row["choices"].tolist()) if c is not None]
        elif isinstance(row["choices"], list) and len(row["choices"]) > 0:
            choices = [f"{i}. {c}" for i, c in enumerate(row["choices"])]
        if not choices:
            choices = ["0. True", "1. False"]
        question = f"{row['question']}\n选项: {', '.join(choices)}，请只输出选项前的数字, 不要输出选项或选项本身"
        return img, question, int(row["answer"]), idx

def collate_fn(batch):
    # batch: list[(img, question, answer, idx)]
    imgs, questions, answers, idxs = zip(*batch)
    # imgs 已经是 list[PIL.Image]（每条样本 1 张图）
    return list(imgs), list(questions), list(answers), list(idxs)

# ----------- DataLoader 参数 -----------
batch_size = 1
num_workers = 4
conv_mode = "llava_v1"
temperature = 0.1
max_new_tokens = 128
top_p = 0.7
num_beams = 4

dataset = SqaDataset(image_data)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn
)

# ----------- 推理 -----------
sample_results = []
all_predictions = []
all_references = []

print("\n=== 开始 batch 推理 ===")
start = time.time()

for micro_batch in tqdm(loader, desc="batch infer"):
    imgs, qs_list, refs, idxs = micro_batch

    # 1. 图像
    image_tensor = process_images(imgs, image_processor, model.config).to(
        device, dtype=torch.float16
    )

    # 2. 文字
    # 2. prompt 构造（关键修复：插入 <image>）
    use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    image_token_se = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if use_im_start_end else DEFAULT_IMAGE_TOKEN
    )

    prompts = []
    for q in qs_list:
        qs = q
        if use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    # 3. tokenize
    input_ids_list = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        for p in prompts
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(device)

    # 3.1 图像尺寸（LLaVA-1.5+ 需要）
    image_sizes = [img.size for img in imgs]
    # 4. generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,      # 新增
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,             # 数字答案很短
            use_cache=True,
        )
    # 5. decode（修复切片）
    raw_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [out.strip() for out in raw_outputs]
    # print(preds)

    # 6. 记录
    for ans, pred, q, idx in zip(refs, preds, qs_list, idxs):
        sample_results.append(
            {
                "sample_id": idx,
                "question": q.split("\n")[0],
                "choices": q.split("\n")[1].replace("选项: ", ""),
                "predicted_answer": pred,
                "correct_answer": ans,
            }
        )
        all_predictions.append(pred)
        all_references.append(ans)

end = time.time()
print(f"\n批推理完成，耗时 {end-start:.2f}s")

# ----------- 评估 & 保存 -----------
metrics = calculate_metrics(all_predictions, all_references)
print("\n=== 评估结果 ===")
print(f"总样本数: {len(image_data)}")
print(f"成功处理样本数: {len(sample_results)}")
print(f"准确率: {metrics['accuracy']}")
print(f"F1分数: {metrics['f1_score']}")

# CSV
out_df = pd.DataFrame(sample_results)
out_path = "/root/autodl-tmp/code/Vqa_ocr/scienceqa_eval_results.csv"
out_df.to_csv(out_path, index=False)
print(f"详细结果已保存至: {out_path}")

# JSON
json.dump(
    {
        "total_samples": len(image_data),
        "processed_samples": len(sample_results),
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "processing_time": end - start,
    },
    open("/root/autodl-tmp/code/Vqa_ocr/result_eval.json", "w", encoding="utf-8"),
    indent=4,
    ensure_ascii=False,
)

# TXT
with open("/root/autodl-tmp/code/Vqa_ocr/result_eval.txt", "w", encoding="utf-8") as f:
    f.write("=== ScienceQA 评估结果 ===\n")
    f.write(f"总样本数: {len(image_data)}\n")
    f.write(f"成功处理样本数: {len(sample_results)}\n")
    f.write(f"准确率: {metrics['accuracy']}\n")
    f.write(f"F1分数: {metrics['f1_score']}\n")
    f.write(f"处理时间: {end-start:.2f}s\n")

print("\n=== 处理完成 ===")