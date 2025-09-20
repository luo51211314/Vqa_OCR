from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

model_path = "/root/autodl-tmp/model/pix2struct_hug"
try:
    model = Pix2StructForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    processor = Pix2StructProcessor.from_pretrained(model_path, local_files_only=True)
    print("模型加载成功！")
except Exception as e:
    print(f"加载失败: {e}")
