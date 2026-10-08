import re
import os
from typing import List, Dict, Any
import wordninja

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"


class SpellingCorrector:
    """基于wordninja的拼写纠错和词分割模块
    
    使用wordninja进行智能词分割，自动识别单词边界
    """
    
    def __init__(self, debug: bool = True, preloaded_model: Dict[str, Any] = None):
        """
        Args:
            debug: 是否输出调试信息
            preloaded_model: 预加载的模型字典（保留接口兼容性）
        """
        self.debug = debug
        self._initialized = True
        
        if self.debug:
            print(f"[拼写纠错] 使用wordninja进行词分割")
    
    def _is_numeric_value(self, text: str) -> bool:
        """检查文本是否为数值（包括小数、百分号等）
        
        Args:
            text: 输入文本
            
        Returns:
            True如果是数值，False否则
        """
        if not text:
            return False
        
        # 去除百分号后检查是否为数字
        cleaned = text.replace('%', '').replace(',', '').strip()
        
        # 检查是否为纯数字或小数
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def _split_by_wordninja(self, text: str) -> str:
        """使用wordninja进行词分割
        
        Args:
            text: 输入文本
            
        Returns:
            分割后的文本
        """
        if not text or len(text) <= 3:
            return text
        
        try:
            # 首先按空格分割，保留原始分隔
            original_words = text.split()
            
            # 对每个词单独处理
            processed_words = []
            for word in original_words:
                # 跳过数值（包括小数、百分号等）
                if self._is_numeric_value(word):
                    processed_words.append(word)
                    continue
                
                # 使用wordninja进行分割
                split_parts = wordninja.split(word)
                processed_words.append(" ".join(split_parts))
            
            return " ".join(processed_words)
                
        except Exception as e:
            if self.debug:
                print(f"[拼写纠错] wordninja分词失败: {e}")
            return text
    
    def correct_text(self, text: str) -> str:
        """对单个文本进行拼写纠错和词分割
        
        Args:
            text: 输入文本
            
        Returns:
            纠错后的文本
        """
        if not text or len(text.strip()) == 0:
            return text
        
        # 使用wordninja进行分词
        corrected = self._split_by_wordninja(text)
        
        return corrected
    
    def correct_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对多个文本块进行拼写纠错
        
        Args:
            blocks: 文本块列表
            
        Returns:
            纠错后的文本块列表
        """
        if not blocks:
            return []
        
        corrected_blocks = []
        correction_count = 0
        
        for block in blocks:
            corrected_block = block.copy()
            
            if "merged_texts" in block:
                # 处理合并的文本
                corrected_texts = []
                for i, text in enumerate(block["merged_texts"]):
                    corrected = self.correct_text(text)
                    if corrected != text:
                        correction_count += 1
                        if self.debug:
                            print(f"  纠错: 'text{i+1}: {text}' -> 'text{i+1}: {corrected}'")
                    corrected_texts.append(corrected)
                
                # 重新构建合并文本
                merged_text = " ".join(f"text{i+1}: {text}" for i, text in enumerate(corrected_texts))
                corrected_block["text"] = merged_text
                corrected_block["merged_texts"] = corrected_texts
            else:
                # 处理单个文本
                original_text = block.get("text", "")
                corrected_text = self.correct_text(original_text)
                if corrected_text != original_text:
                    correction_count += 1
                    if self.debug:
                        print(f"  纠错: '{original_text}' -> '{corrected_text}'")
                corrected_block["text"] = corrected_text
            
            corrected_blocks.append(corrected_block)
        
        if self.debug:
            print(f"[拼写纠错] 完成，共纠错 {correction_count} 个文本块")
        
        return corrected_blocks
