import os
from typing import List, Dict, Any, Optional
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"


class StagePromptGenerator:
    """阶段性prompt生成器
    
    根据问题类型生成阶段性指导的prompt，提高模型回答的准确性
    支持二阶段推理：
    阶段1：去除无关block
    阶段2：使用step prompt进行推理
    """
    
    # 阶段1：去除无关block的prompt模板
    STAGE1_FILTER_TEMPLATES = {
        "counting": "Keep ONLY blocks that contain actual item names with their numerical values. Remove titles, sources, axes labels, metadata, and any descriptive text.",
        "calculation": "Keep ONLY blocks that contain numerical values needed for calculation. Remove titles, sources, axes labels, and any non-numerical descriptive text.",
        "comparison": "Keep ONLY blocks that contain the items mentioned in the question with their values. Remove all other irrelevant blocks including titles, sources, and metadata.",
        "difference": "Keep ONLY blocks that contain the numerical values needed for calculation. Remove titles, sources, axes labels, and any non-numerical descriptive text.",
        "retrieval": "Keep ONLY blocks that contain the specific information needed to answer the question. Remove titles, sources, and any irrelevant metadata.",
        "trend": "Keep ONLY blocks that contain data points with timestamps or sequential values. Remove titles, sources, and static metadata.",
        "comprehension": "Keep ONLY blocks that contain information directly relevant to answering the question. Remove titles, sources, axes labels, metadata, and any descriptive text.",
        "default": "Keep ONLY blocks that contain actual item names with their values. Remove titles, sources, axes labels, metadata, and any descriptive text that does not contain specific data points."
    }
    
    # 阶段2：step prompt模板
    STAGE2_STEP_TEMPLATES = {
        "counting": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'index', 'price', 'source', 'data', 'world', 'research', 'commodity', 'since', 'by', or pure numbers. Focus ONLY on blocks containing single item names (1-2 words) followed by a number.
Step 1: Extract ONLY the item names from the relevant blocks.
Step 2: Count the number of distinct items.
Step 3: Answer with ONLY the numerical count.""",
        
        "calculation": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'index', 'price', 'source', 'data', 'world', 'research', 'commodity', 'since', 'by', or pure numbers. Focus ONLY on blocks containing numerical values or data needed for calculation.
Step 1: Extract the relevant numerical values from the filtered blocks.
Step 2: Perform the required calculation.
Step 3: Answer with ONLY the numerical result.""",
        
        "comparison": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'index', 'price', 'source', 'data', 'world', 'research', 'commodity', 'since', 'by', or pure numbers. Focus ONLY on blocks containing the specific items mentioned in the question with their values.
Step 1: Extract the values for the items being compared.
Step 2: Compare the values.
Step 3: Answer with ONLY 'Yes' or 'No'.""",
        
        "difference": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'index', 'price', 'source', 'data', 'world', 'research', 'commodity', 'since', 'by', or pure numbers. Focus ONLY on blocks containing the specific items mentioned in the question with their numerical values.
Step 1: Extract the two values mentioned in the question.
Step 2: Calculate the difference (higher value - lower value).
Step 3: Answer with ONLY the numerical result.""",
        
        "retrieval": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'index', 'price', 'source', 'data', 'world', 'research', 'commodity', 'since', 'by', or pure numbers. Focus ONLY on blocks containing item names with their values.
Step 1: Extract all item names and their corresponding values.
Step 2: Identify the item with the required property (highest, lowest, etc.).
Step 3: Answer with ONLY the item name.""",
        
        "trend": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'source', 'data', 'world', 'research', or metadata. Focus ONLY on blocks containing data points with timestamps or sequential values.
Step 1: Extract the data points in chronological order.
Step 2: Analyze the trend (increasing, decreasing, or stable).
Step 3: Answer the question based on the trend analysis.""",
        
        "comprehension": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'index', 'price', 'source', 'data', 'world', 'research', 'commodity', 'since', 'by', or pure numbers. Focus ONLY on blocks containing actual data.
Step 1: Extract the relevant information from the filtered blocks.
Step 2: Analyze the information to answer the question.
Step 3: Provide a concise answer.""",
        
        "default": """Step 0: Ignore blocks with long text (>10 words), blocks containing words like 'index', 'price', 'source', 'data', 'world', 'research', 'commodity', 'since', 'by', or pure numbers. Focus ONLY on blocks containing actual data.
Step 1: Extract the relevant information from the filtered blocks.
Step 2: Analyze the information to answer the question.
Step 3: Provide a concise answer."""
    }
    
    # 兼容旧版本：单阶段prompt模板
    QUESTION_TEMPLATES = {
        "counting": {
            "template": "Step 0: Ignore any blocks that contain irrelevant information such as titles, sources, axes labels, or metadata. Focus ONLY on blocks that contain actual item names or data values. Step 1: Look at each relevant block and extract ONLY the item names that actually appear. Step 2: List these items exactly as they appear. Step 3: Count them. Step 4: Answer with ONLY the number.\n\nQuestion: {question}\n\nBlocks: {blocks}"
        },
        "calculation": {
            "template": "Step 0: Ignore any blocks that contain irrelevant information such as titles, sources, axes labels, metadata, or long descriptive text. Focus ONLY on blocks that contain SHORT item names (1-3 words) followed by numerical values. Step 1: Extract ONLY the item names (not the numbers) from the relevant blocks. Step 2: Count the number of distinct items. Step 3: Answer with ONLY the numerical count.\n\nQuestion: {question}\n\nBlocks: {blocks}"
        },
        "difference": {
            "template": "Step 0: Ignore any blocks that contain irrelevant information such as titles, sources, axes labels, or metadata. Focus ONLY on blocks that contain the specific values mentioned in the question. Step 1: Extract the two values mentioned in the question from the relevant blocks. Step 2: Calculate the difference (first value - second value). Step 3: Answer with ONLY the number.\n\nQuestion: {question}\n\nBlocks: {blocks}"
        },
        "comparison": {
            "template": "Step 0: Ignore any blocks that contain irrelevant information such as titles, sources, axes labels, or metadata. Focus ONLY on blocks that contain the values for comparison. Step 1: Extract the values for comparison from the relevant blocks. Step 2: Compare them. Step 3: Answer with ONLY 'Yes' or 'No'.\n\nQuestion: {question}\n\nBlocks: {blocks}"
        },
        "trend": {
            "template": "Step 0: Ignore any blocks that contain irrelevant information such as titles, sources, axes labels, or metadata. Focus ONLY on blocks that contain actual data points or values. Step 1: Extract the data points from the relevant blocks. Step 2: Identify the trend/pattern. Step 3: Answer the question.\n\nQuestion: {question}\n\nBlocks: {blocks}"
        },
        "retrieval": {
            "template": "Step 0: Ignore any blocks that contain irrelevant information such as titles, sources, axes labels, or metadata. Focus ONLY on blocks that contain information directly relevant to answering the question. Step 1: Extract the relevant information from the filtered blocks. Step 2: Answer the question directly.\n\nQuestion: {question}\n\nBlocks: {blocks}"
        },
        "comprehension": {
            "template": "Step 0: Ignore any blocks that contain irrelevant information such as titles, sources, axes labels, or metadata. Focus ONLY on blocks that contain information directly relevant to answering the question. Step 1: Extract the relevant information from the filtered blocks. Step 2: Answer the question directly.\n\nQuestion: {question}\n\nBlocks: {blocks}"
        }
    }
    
    def __init__(self, debug: bool = True, use_two_stage: bool = True):
        """
        Args:
            debug: 是否输出调试信息
            use_two_stage: 是否使用二阶段推理（阶段1过滤 + 阶段2推理）
        """
        self.debug = debug
        self.use_two_stage = use_two_stage
    
    def blocks_to_string(self, blocks: List[Dict[str, Any]]) -> str:
        """将blocks列表转换为字符串格式"""
        block_texts = []
        for i, block in enumerate(blocks):
            text = block.get("text", "")
            block_texts.append(f"block{i+1}:{{text: {text}}}")
        return " ".join(block_texts)
    
    def string_to_blocks(self, blocks_str: str) -> List[Dict[str, Any]]:
        """将字符串格式转换回blocks列表"""
        blocks = []
        pattern = r'block\d+:\{text:\s*([^}]+)\}'
        matches = re.findall(pattern, blocks_str)
        for match in matches:
            text = match.strip()
            # 去除可能的重复"text: "前缀
            if text.startswith("text: "):
                text = text[6:].strip()
            blocks.append({"text": text})
        return blocks
    
    def generate_stage1_prompt(self, question: str, question_type: str, blocks_str: str) -> str:
        """生成阶段1的prompt（去除无关block）
        
        Args:
            question: 问题文本
            question_type: 问题类型
            blocks_str: blocks字符串
            
        Returns:
            阶段1的prompt
        """
        # 获取过滤指令
        instruction = self.STAGE1_FILTER_TEMPLATES.get(
            question_type, 
            self.STAGE1_FILTER_TEMPLATES["default"]
        )
        
        prompt = f"""Please analyze the blocks below and remove any blocks that are irrelevant to answering the question. {instruction}

Question: {question}

Blocks: {blocks_str}

Output ONLY the filtered blocks in the same format as input (block1:{{text: ...}} block2:{{text: ...}} ...). Do not include any explanation or additional text."""
        
        return prompt
    
    def generate_stage2_prompt(self, question: str, question_type: str, blocks_str: str) -> str:
        """生成阶段2的prompt（使用step prompt推理）
        
        Args:
            question: 问题文本
            question_type: 问题类型
            blocks_str: blocks字符串（已过滤）
            
        Returns:
            阶段2的prompt
        """
        # 获取step prompt
        step_prompt = self.STAGE2_STEP_TEMPLATES.get(
            question_type,
            self.STAGE2_STEP_TEMPLATES["default"]
        )
        
        prompt = f"""{step_prompt}

Question: {question}

Blocks: {blocks_str}

Based on the above information, please answer the following question: {question}
Please provide a complete and detailed answer, including all relevant information. If the question involves numerical values, please output in Arabic numeral form (e.g., 1, 2, 3) instead of English words (one, two, three). Please answer in the same sentence structure as the question: for yes/no questions, answer with 'Yes' or 'No'; for counting questions, answer with the number; for comparison questions, state the specific values and calculate the difference."""
        
        return prompt
    
    def generate_stage_prompt(self, question: str, blocks: List[Dict[str, Any]], question_type: str) -> str:
        """生成阶段性指导prompt（兼容旧版本，单阶段）
        
        Args:
            question: 问题文本
            blocks: 文本块列表
            question_type: 问题类型（来自question_classifier）
            
        Returns:
            生成的阶段性prompt
        """
        # 使用传入的问题类型，如果不存在则使用默认类型
        if question_type not in self.QUESTION_TEMPLATES:
            if self.debug:
                print(f"[阶段性Prompt] 未知问题类型: {question_type}，使用默认类型: retrieval")
            question_type = "retrieval"
        
        # 构建blocks文本
        blocks_str = self.blocks_to_string(blocks)
        
        # 生成prompt
        template = self.QUESTION_TEMPLATES[question_type]["template"]
        prompt = template.format(
            question=question,
            blocks=blocks_str
        )
        
        if self.debug:
            print(f"[阶段性Prompt] 使用问题类型: {question_type}")
            print(f"[阶段性Prompt] 生成的完整Prompt:")
            print(prompt)
            print()
        
        return prompt
    
    def generate_two_stage_prompts(self, question: str, blocks: List[Dict[str, Any]], question_type: str) -> Dict[str, str]:
        """生成二阶段prompt
        
        Args:
            question: 问题文本
            blocks: 文本块列表
            question_type: 问题类型（来自question_classifier）
            
        Returns:
            包含两个阶段prompt的字典：{"stage1": ..., "stage2": ...}
        """
        # 标准化问题类型
        if question_type not in self.STAGE1_FILTER_TEMPLATES:
            if self.debug:
                print(f"[二阶段Prompt] 未知问题类型: {question_type}，使用默认类型")
            question_type = "default"
        
        # 构建blocks字符串
        blocks_str = self.blocks_to_string(blocks)
        
        # 生成两个阶段的prompt
        stage1_prompt = self.generate_stage1_prompt(question, question_type, blocks_str)
        stage2_prompt = self.generate_stage2_prompt(question, question_type, blocks_str)
        
        if self.debug:
            print(f"[二阶段Prompt] 问题类型: {question_type}")
            print(f"[二阶段Prompt] 原始blocks数量: {len(blocks)}")
            print()
        
        return {
            "stage1": stage1_prompt,
            "stage2": stage2_prompt,
            "question_type": question_type,
            "original_blocks": blocks_str
        }
