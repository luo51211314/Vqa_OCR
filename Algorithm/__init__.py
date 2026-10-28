"""Algorithm 包。

注意：子模块（dbscan_clustering / ocr_pipeline 等）依赖较重的第三方库
（如 hdbscan、transformers）且只被可选的 OCR pipeline（USE_PIPELINE=yes，
属于 L2 补充实验路径）使用。主表推理路径（reading_order / ocr_filters）
绝不应被这些可选依赖拖累，因此这里采用惰性导入：
``from Algorithm import OCRPipeline`` 等旧写法仍然可用，但只有真正访问
该名字时才加载对应模块。
"""
import importlib

__all__ = [
    "DBSCANClustering",
    "SpellingCorrector",
    "QuestionClassifier",
    "StagePromptGenerator",
    "OCRPipeline",
]

# 导出名 -> 所在子模块
_LAZY = {
    "DBSCANClustering": "dbscan_clustering",
    "SpellingCorrector": "spelling_correction",
    "QuestionClassifier": "question_classifier",
    "StagePromptGenerator": "stage_prompt_generator",
    "OCRPipeline": "ocr_pipeline",
}


def __getattr__(name):
    """PEP 562：按需导入重子模块，避免 import Algorithm 即拉起 hdbscan。"""
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
