"""跨模型共享的推理配置。

论文修订版要求：除各模型官方 chat template / image token 包装（属于模型推理
接口，无法统一）外，输入给模型的指令内容必须逐字一致。因此三个 loader 的
``prompt_suffix`` 统一从本文件获取；解码参数（temperature / top_p /
num_beams / max_new_tokens）也在此集中定义，避免跨模型横向比较时引入混淆
变量。

只允许在一处修改，不要再在各个 loader 内硬编码后缀。
"""

# 与指标类型对应的统一后缀（三个模型共用，逐字一致）
PROMPT_SUFFIX_BY_METRIC = {
    "anls": "\nAnswer briefly.",
    "relaxed_accuracy": "\nAnswer briefly. If digits in answer, use Arabic numerals.",
    "relaxed_accuracy_80": "\nAnswer briefly. If digits in answer, use Arabic numerals.",
    "bleu": "\nAnswer in 1-3 sentences. If digits in answer, use Arabic numerals.",
    "weighted": "\nAnswer in 1-3 sentences. If digits in answer, use Arabic numerals.",
}

# 统一的确定性解码配置（greedy；num_beams=1）
GENERATION_CONFIG_BY_METRIC = {
    "anls":               {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "num_beams": 1, "max_new_tokens": 64},
    "relaxed_accuracy":   {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "num_beams": 1, "max_new_tokens": 128},
    "relaxed_accuracy_80":{"temperature": 0.0, "top_p": 1.0, "do_sample": False, "num_beams": 1, "max_new_tokens": 128},
    "bleu":               {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "num_beams": 1, "max_new_tokens": 128},
    "weighted":           {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "num_beams": 1, "max_new_tokens": 128},
}


def get_prompt_suffix(metric_type: str) -> str:
    """返回某指标下三个模型统一使用的 prompt 后缀。"""
    return PROMPT_SUFFIX_BY_METRIC.get(metric_type, PROMPT_SUFFIX_BY_METRIC["anls"])


def get_generation_config(metric_type: str) -> dict:
    """返回某指标下统一的确定性生成配置（拷贝，避免调用方污染常量）。"""
    base = GENERATION_CONFIG_BY_METRIC.get(metric_type, GENERATION_CONFIG_BY_METRIC["anls"])
    return dict(base)
