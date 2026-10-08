from .loader_base import BaseModelLoader

# 模型注册表 - 使用懒加载
MODEL_LOADERS = {
    "llava": "loader_llava.LLaVALoader",
    "qwen": "loader_qwen.QwenLoader",
    "blip": "loader_blip.BlipLoader",
    "mplug": "loader_mplug.MplugLoader",
}

def get_model_loader(model_name):
    """获取模型加载器"""
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_LOADERS.keys())}")
    
    # 懒加载对应的模型加载器
    loader_path = MODEL_LOADERS[model_name]
    module_name, class_name = loader_path.split('.')
    
    # 动态导入模块
    import importlib
    module = importlib.import_module(f".{module_name}", __name__)
    loader_class = getattr(module, class_name)
    
    return loader_class()
