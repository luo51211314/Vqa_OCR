#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 llava 导入问题
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("调试 llava 导入")
print("=" * 80)

# 尝试直接导入
print("\n1. 尝试直接从 models.llava.llava.model.language_model.llava_llama 导入...")
try:
    from models.llava.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    print("✓ 成功导入 LlavaLlamaForCausalLM")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

# 查看一下模型的路径
print("\n2. 查看相关文件路径:")
try:
    llava_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "llava", "llava", "model")
    print(f"llava model 目录: {llava_model_dir}")
    
    llava_llama_file = os.path.join(llava_model_dir, "language_model", "llava_llama.py")
    print(f"llava_llama.py 文件: {llava_llama_file}")
    print(f"文件存在: {os.path.exists(llava_llama_file)}")
    
    if os.path.exists(llava_llama_file):
        with open(llava_llama_file, "r", encoding="utf-8") as f:
            print("前50行内容:")
            for i, line in enumerate(f):
                if i >= 50:
                    break
                print(f"  {i+1}: {line.rstrip()}")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
