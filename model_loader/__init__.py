from .loader_llava import LLaVALoader
from .loader_qwen import QwenLoader
from .loader_base import BaseModelLoader

# 模型注册表
MODEL_LOADERS = {
    "llava": LLaVALoader,
    "qwen": QwenLoader,
}

def get_model_loader(model_name):
    """获取模型加载器"""
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_LOADERS.keys())}")
    return MODEL_LOADERS[model_name]()