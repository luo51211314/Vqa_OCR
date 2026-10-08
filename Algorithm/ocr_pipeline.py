import os
from typing import List, Dict, Any, Optional, Tuple
import time
import re

from dbscan_clustering import DBSCANClustering
from spelling_correction import SpellingCorrector
from question_classifier import QuestionClassifier
from stage_prompt_generator import StagePromptGenerator


class OCRPipeline:
    """OCR处理流水线
    
    整合处理步骤：
    1. BERT问题分类
    2. OCR粘连纠错
    3. DBSCAN空间密度聚类（含词性分析）
    4. 二阶段推理（可选）：
       - 阶段1：去除无关block
       - 阶段2：使用step prompt进行推理
    
    支持消融实验：可以跳过任意步骤
    """
    
    def __init__(
        self,
        dbscan_eps: float = 30.0,
        debug: bool = True,
        use_clustering: bool = True,
        use_spelling_correction: bool = True,
        use_question_classification: bool = True,
        use_stage_prompt: bool = False,
        use_two_stage: bool = False,
        preloaded_models: Dict[str, Any] = None
    ):
        """
        Args:
            dbscan_eps: DBSCAN的邻域半径
            debug: 是否输出调试信息
            use_clustering: 是否使用聚类步骤
            use_spelling_correction: 是否使用拼写纠错步骤
            use_question_classification: 是否使用问题分类步骤
            use_stage_prompt: 是否使用阶段性prompt生成
            use_two_stage: 是否使用二阶段推理（阶段1过滤 + 阶段2推理）
            preloaded_models: 预加载的模型字典，包含BERT模型和VLM模型
        """
        self.debug = debug
        self.preloaded_models = preloaded_models or {}
        
        self.use_clustering = use_clustering
        self.use_spelling_correction = use_spelling_correction
        self.use_question_classification = use_question_classification
        self.use_stage_prompt = use_stage_prompt
        self.use_two_stage = use_two_stage
        
        self.clustering = DBSCANClustering(eps=dbscan_eps, debug=debug) if use_clustering else None
        
        # 拼写纠错：创建新实例
        if use_spelling_correction:
            self.spelling_corrector = SpellingCorrector(debug=debug)
        else:
            self.spelling_corrector = None
        
        # 问题分类：使用预加载模型或创建新实例
        if use_question_classification:
            if "question_classifier" in self.preloaded_models:
                self.question_classifier = QuestionClassifier(
                    debug=debug,
                    preloaded_model=self.preloaded_models["question_classifier"]
                )
            else:
                self.question_classifier = QuestionClassifier(debug=debug)
        else:
            self.question_classifier = None
        
        # 阶段性prompt生成器
        self.stage_prompt_generator = StagePromptGenerator(debug=debug, use_two_stage=use_two_stage) if use_stage_prompt else None
        
        # VLM模型（用于二阶段推理）
        self.vlm_model = self.preloaded_models.get("vlm_model", None)
        self.vlm_processor = self.preloaded_models.get("vlm_processor", None)
        
        if debug:
            print("\n" + "="*60)
            print("[OCR流水线] 初始化完成")
            print(f"  - 聚类步骤: {'启用' if use_clustering else '禁用'}")
            print(f"  - 拼写纠错: {'启用' if use_spelling_correction else '禁用'}")
            print(f"  - 问题分类: {'启用' if use_question_classification else '禁用'}")
            print(f"  - 阶段性Prompt: {'启用' if use_stage_prompt else '禁用'}")
            print(f"  - 二阶段推理: {'启用' if use_two_stage else '禁用'}")
            if self.preloaded_models:
                print(f"  - 使用预加载模型: {list(self.preloaded_models.keys())}")
            print("="*60 + "\n")
    
    def process(
        self,
        ocr_texts: List[Dict[str, Any]],
        question: str = None,
        dataset_type: str = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """执行完整的OCR处理流水线
        
        Args:
            ocr_texts: OCR识别结果列表
            question: 用户问题
            dataset_type: 数据集类型（如mydatavqa的问题类型字段）
            
        Returns:
            (处理后的块列表, 处理统计信息)
        """
        start_time = time.time()
        
        stats = {
            "original_blocks": len(ocr_texts),
            "steps": {},
            "total_time": 0.0
        }
        
        current_blocks = ocr_texts
        if not current_blocks:
            return [], stats
        
        if self.debug:
            print("\n" + "="*60)
            print("[OCR流水线] 开始处理")
            print(f"  原始OCR块数: {len(current_blocks)}")
            if question:
                print(f"  问题: {question[:50]}...")
            if dataset_type:
                print(f"  数据集类型: {dataset_type}")
            print("="*60)
        
        q_type = "comprehension"
        q_confidence = 1.0
        
        if self.use_question_classification and self.question_classifier:
            step_start = time.time()
            q_type, q_confidence = self.question_classifier.classify_with_dataset_type(
                question, dataset_type
            )
            stats["steps"]["question_classification"] = {
                "type": q_type,
                "confidence": q_confidence,
                "time": time.time() - step_start
            }
            if self.debug:
                print(f"\n[步骤: 问题分类] 类型={q_type}, 置信度={q_confidence:.3f}")
        
        if self.use_spelling_correction and self.spelling_corrector:
            step_start = time.time()
            current_blocks = self.spelling_corrector.correct_blocks(current_blocks)
            stats["steps"]["spelling_correction"] = {
                "time": time.time() - step_start
            }
            if self.debug:
                print(f"\n[步骤: 拼写纠错] 完成")
        
        if self.use_clustering and self.clustering:
            step_start = time.time()
            current_blocks = self.clustering.cluster(current_blocks)
            stats["steps"]["clustering"] = {
                "blocks_after": len(current_blocks),
                "time": time.time() - step_start
            }
            if self.debug:
                print(f"\n[步骤: DBSCAN聚类] 聚类后块数={len(current_blocks)}")
        
        # 阶段性prompt生成（单阶段）
        stage_prompt = None
        if self.use_stage_prompt and self.stage_prompt_generator and question and not self.use_two_stage:
            step_start = time.time()
            stage_prompt = self.stage_prompt_generator.generate_stage_prompt(question, current_blocks, q_type)
            stats["steps"]["stage_prompt"] = {
                "time": time.time() - step_start
            }
            if self.debug:
                print(f"\n[步骤: 阶段性Prompt生成] 完成")
        
        stats["final_blocks"] = len(current_blocks)
        stats["total_time"] = time.time() - start_time
        stats["question_type"] = q_type
        stats["question_confidence"] = q_confidence
        stats["stage_prompt"] = stage_prompt
        
        if self.debug:
            print("\n" + "="*60)
            print("[OCR流水线] 处理完成")
            print(f"  最终块数: {stats['final_blocks']}")
            print(f"  总耗时: {stats['total_time']:.3f}s")
            print("="*60 + "\n")
        
        return current_blocks, stats
    
    def two_stage_inference(
        self,
        question: str,
        blocks: List[Dict[str, Any]],
        question_type: str,
        image = None
    ) -> Tuple[str, Dict[str, Any], str]:
        """执行二阶段推理
        
        阶段1：去除无关block
        阶段2：使用step prompt进行推理
        
        Args:
            question: 用户问题
            blocks: 处理后的块列表
            question_type: 问题类型
            image: 可选的图像数据
            
        Returns:
            (最终答案, 推理统计信息)
        """
        if not self.vlm_model or not self.vlm_processor:
            raise ValueError("二阶段推理需要预加载VLM模型和processor")
        
        stats = {
            "stage1": {},
            "stage2": {},
            "total_time": 0.0
        }
        start_time = time.time()
        
        # 生成二阶段prompts
        two_stage_prompts = self.stage_prompt_generator.generate_two_stage_prompts(
            question, blocks, question_type
        )
        
        # ========== 阶段1：去除无关block ==========
        if self.debug:
            print("\n" + "="*60)
            print("[二阶段推理] 阶段1：去除无关block")
            print("="*60)
        
        stage1_start = time.time()
        stage1_prompt = two_stage_prompts["stage1"]
        
        messages_stage1 = [
            {"role": "system", "content": "You are a helpful assistant. Filter out irrelevant information."},
            {"role": "user", "content": stage1_prompt}
        ]
        
        text_stage1 = self.vlm_processor.apply_chat_template(
            messages_stage1, tokenize=False, add_generation_prompt=True
        )
        
        # 如果有图像，加入图像输入
        if image is not None:
            inputs_stage1 = self.vlm_processor(
                text=[text_stage1],
                images=[image],
                return_tensors="pt"
            ).to(self.vlm_model.device)
        else:
            inputs_stage1 = self.vlm_processor(
                text=[text_stage1],
                return_tensors="pt"
            ).to(self.vlm_model.device)
        
        outputs_stage1 = self.vlm_model.generate(
            **inputs_stage1,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.vlm_processor.tokenizer.pad_token_id,
            eos_token_id=self.vlm_processor.tokenizer.eos_token_id
        )
        
        full_output_stage1 = self.vlm_processor.batch_decode(
            outputs_stage1, skip_special_tokens=True
        )[0]
        input_text_stage1 = text_stage1.strip()
        
        # 提取生成的内容
        if input_text_stage1 in full_output_stage1:
            filtered_blocks_str = full_output_stage1[len(input_text_stage1):].strip()
        else:
            input_length = inputs_stage1['input_ids'].shape[1]
            generated_ids = outputs_stage1[0][input_length:]
            filtered_blocks_str = self.vlm_processor.batch_decode(
                [generated_ids], skip_special_tokens=True
            )[0]
        
        # 解析过滤后的blocks
        filtered_blocks = self.stage_prompt_generator.string_to_blocks(filtered_blocks_str)
        
        # 保存原始blocks字符串
        original_blocks_str = self.stage_prompt_generator.blocks_to_string(blocks)
        
        stats["stage1"]["time"] = time.time() - stage1_start
        stats["stage1"]["original_blocks_count"] = len(blocks)
        stats["stage1"]["filtered_blocks_count"] = len(filtered_blocks)
        stats["stage1"]["filtered_blocks_str"] = filtered_blocks_str
        stats["original_blocks_str"] = original_blocks_str
        
        if self.debug:
            print(f"  原始blocks数量: {len(blocks)}")
            print(f"  过滤后blocks数量: {len(filtered_blocks)}")
            print(f"  阶段1耗时: {stats['stage1']['time']:.3f}s")
            print(f"  [调试] 原始blocks字符串: {original_blocks_str[:200]}..." if len(original_blocks_str) > 200 else f"  [调试] 原始blocks字符串: {original_blocks_str}")
            print(f"  [调试] 过滤后blocks字符串: {filtered_blocks_str[:200]}..." if len(filtered_blocks_str) > 200 else f"  [调试] 过滤后blocks字符串: {filtered_blocks_str}")
        
        # ========== 阶段2：使用step prompt进行推理 ==========
        if self.debug:
            print("\n" + "="*60)
            print("[二阶段推理] 阶段2：使用step prompt进行推理")
            print("="*60)
        
        stage2_start = time.time()
        stage2_prompt = two_stage_prompts["stage2"]
        
        # 使用过滤后的blocks替换原始blocks
        stage2_prompt_with_filtered = stage2_prompt.replace(
            two_stage_prompts["original_blocks"],
            filtered_blocks_str
        )
        
        messages_stage2 = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided information."},
            {"role": "user", "content": stage2_prompt_with_filtered}
        ]
        
        text_stage2 = self.vlm_processor.apply_chat_template(
            messages_stage2, tokenize=False, add_generation_prompt=True
        )
        
        # 如果有图像，加入图像输入
        if image is not None:
            inputs_stage2 = self.vlm_processor(
                text=[text_stage2],
                images=[image],
                return_tensors="pt"
            ).to(self.vlm_model.device)
        else:
            inputs_stage2 = self.vlm_processor(
                text=[text_stage2],
                return_tensors="pt"
            ).to(self.vlm_model.device)
        
        outputs_stage2 = self.vlm_model.generate(
            **inputs_stage2,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.vlm_processor.tokenizer.pad_token_id,
            eos_token_id=self.vlm_processor.tokenizer.eos_token_id
        )
        
        full_output_stage2 = self.vlm_processor.batch_decode(
            outputs_stage2, skip_special_tokens=True
        )[0]
        input_text_stage2 = text_stage2.strip()
        
        # 提取生成的答案
        if input_text_stage2 in full_output_stage2:
            answer = full_output_stage2[len(input_text_stage2):].strip()
        else:
            input_length = inputs_stage2['input_ids'].shape[1]
            generated_ids = outputs_stage2[0][input_length:]
            answer = self.vlm_processor.batch_decode(
                [generated_ids], skip_special_tokens=True
            )[0]
        
        stats["stage2"]["time"] = time.time() - stage2_start
        stats["stage2"]["answer"] = answer
        
        stats["total_time"] = time.time() - start_time
        
        if self.debug:
            print(f"  模型回答: {answer}")
            print(f"  阶段2耗时: {stats['stage2']['time']:.3f}s")
            print(f"  总耗时: {stats['total_time']:.3f}s")
            print("="*60 + "\n")
        
        # 返回答案、统计信息和完整prompt
        return answer, stats, stage2_prompt_with_filtered
    
    def to_prompt(self, blocks: List[Dict[str, Any]], stats: Dict[str, Any] = None, max_length: int = 3000) -> str:
        """将处理后的块转换为prompt格式
        
        Args:
            blocks: 处理后的块列表
            stats: 处理统计信息
            max_length: 最大prompt长度
            
        Returns:
            prompt字符串
        """
        if not blocks:
            return "the ocr text: not found ocr text"
        
        block_texts = []
        for i, block in enumerate(blocks):
            text = block.get("text", "")
            minx = round(block.get("minx", 0.0), 2)
            miny = round(block.get("miny", 0.0), 2)
            maxx = round(block.get("maxx", 100.0), 2)
            maxy = round(block.get("maxy", 30.0), 2)
            block_texts.append(f"block{i+1}:{{pos:({minx},{miny}),({maxx},{maxy}), {text}}}")
        
        full_prompt = " ".join(block_texts)
        
        if len(full_prompt) > max_length:
            full_prompt = full_prompt[:max_length] + "..."
        
        return f"format: {full_prompt}"
    
    def process_and_to_prompt(
        self,
        ocr_texts: List[Dict[str, Any]],
        question: str = None,
        dataset_type: str = None,
        max_length: int = 3000,
        image = None
    ) -> Tuple[str, Dict[str, Any]]:
        """处理OCR并生成prompt
        
        Args:
            ocr_texts: OCR识别结果列表
            question: 用户问题
            dataset_type: 数据集类型
            max_length: 最大prompt长度
            image: 可选的图像数据（用于二阶段推理）
            
        Returns:
            (prompt字符串或答案, 处理统计信息)
        """
        blocks, stats = self.process(ocr_texts, question, dataset_type)
        
        # 如果使用二阶段推理，执行推理并返回完整信息
        if self.use_two_stage and self.vlm_model and question:
            q_type = stats.get("question_type", "comprehension")
            answer, inference_stats, full_prompt = self.two_stage_inference(
                question, blocks, q_type, image
            )
            stats["two_stage_inference"] = inference_stats
            stats["full_prompt"] = full_prompt
            # 在二阶段模式下，返回完整prompt和stats，而不是答案
            return full_prompt, stats
        
        # 如果使用阶段性prompt（单阶段），直接返回生成的阶段性prompt
        if self.use_stage_prompt and stats.get("stage_prompt"):
            return stats["stage_prompt"], stats
        
        # 否则使用传统的prompt格式
        prompt = self.to_prompt(blocks, stats, max_length)
        
        return prompt, stats


def create_pipeline(
    mode: str = "full",
    dbscan_eps: float = 30.0,
    debug: bool = True,
    preloaded_models: Dict[str, Any] = None
) -> OCRPipeline:
    """创建不同配置的OCR流水线
    
    Args:
        mode: 流水线模式
            - "full": 完整流水线（默认启用二阶段推理）
            - "no_two_stage": 禁用二阶段推理（使用单阶段prompt）
            - "no_clustering": 不使用聚类步骤
            - "no_spelling": 不使用拼写纠错步骤
            - "no_classification": 不使用问题分类步骤
            - "no_all": 不使用任何处理步骤
            - "no_stage": 不使用阶段性prompt
        dbscan_eps: DBSCAN的邻域半径
        debug: 是否输出调试信息
        preloaded_models: 预加载的模型字典，包含BERT模型和VLM模型
        
    Returns:
        OCRPipeline实例
    """
    mode_config = {
        "full": {
            "use_clustering": True,
            "use_spelling_correction": True,
            "use_question_classification": True,
            "use_stage_prompt": True,
            "use_two_stage": True  # 默认启用二阶段推理
        },
        "no_two_stage": {
            "use_clustering": True,
            "use_spelling_correction": True,
            "use_question_classification": True,
            "use_stage_prompt": True,
            "use_two_stage": False  # 禁用二阶段推理，使用单阶段prompt
        },
        "no_clustering": {
            "use_clustering": False,
            "use_spelling_correction": True,
            "use_question_classification": True,
            "use_stage_prompt": True,
            "use_two_stage": True
        },
        "no_spelling": {
            "use_clustering": True,
            "use_spelling_correction": False,
            "use_question_classification": True,
            "use_stage_prompt": True,
            "use_two_stage": True
        },
        "no_all": {
            "use_clustering": False,
            "use_spelling_correction": False,
            "use_question_classification": False,
            "use_stage_prompt": True,
            "use_two_stage": True
        },
        "no_stage": {
            "use_clustering": True,
            "use_spelling_correction": True,
            "use_question_classification": True,
            "use_stage_prompt": False,
            "use_two_stage": False
        }
    }
    
    if mode not in mode_config:
        print(f"警告: 未知模式 '{mode}'，使用 'full' 模式")
        mode = "full"
    
    config = mode_config[mode]
    
    return OCRPipeline(
        dbscan_eps=dbscan_eps,
        debug=debug,
        preloaded_models=preloaded_models,
        **config
    )
