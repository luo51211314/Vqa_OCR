import importlib
import os
from typing import Dict, List, Any, Optional, Tuple
from .base_expert import BaseExpert

class ExpertManager:
    """专家模块管理器"""
    
    def __init__(self):
        self.experts: Dict[str, BaseExpert] = {}
        self.available_experts = self._discover_experts()
        self.ocr_pipeline = None
        self.preloaded_models: Dict[str, Any] = {}  # 预加载的BERT模型
    
    def set_preloaded_models(self, models: Dict[str, Any]):
        """设置预加载的BERT模型
        
        Args:
            models: 预加载的模型字典，包含question_classifier, spelling_corrector, content_compressor
        """
        self.preloaded_models = models
        print(f"[ExpertManager] 接收到 {len(models)} 个预加载模型")
        
        # 将预加载模型传递给OCR专家
        ocr_expert = self.get_expert("ocr")
        if ocr_expert and hasattr(ocr_expert, 'set_preloaded_models'):
            ocr_expert.set_preloaded_models(models)
            print("[ExpertManager] 预加载模型已传递给OCR专家")
    
    def _discover_experts(self) -> List[str]:
        """自动发现可用的专家模块"""
        expert_dir = os.path.dirname(__file__)
        experts = []
        
        for file in os.listdir(expert_dir):
            if file.endswith("_expert.py") and file != "base_expert.py":
                expert_name = file[:-10]  # 移除'_expert.py'
                experts.append(expert_name)
        
        return experts
    
    def register_expert(self, name: str, expert: BaseExpert):
        """注册专家模块"""
        self.experts[name] = expert
    
    def get_expert(self, name: str) -> Optional[BaseExpert]:
        """获取专家模块实例"""
        return self.experts.get(name)
    
    def initialize_expert(self, name: str, model_path: Optional[str] = None, **kwargs):
        """初始化特定专家模块"""
        if name not in self.available_experts:
            raise ValueError(f"专家模块 {name} 不可用，可用模块: {self.available_experts}")
        
        # 动态导入专家模块
        module_name = f"expert.{name}_expert"
        try:
            module = importlib.import_module(module_name)
            expert_class = getattr(module, f"{name.capitalize()}Expert")
            expert_instance = expert_class()
            expert_instance.initialize(model_path, **kwargs)
            self.register_expert(name, expert_instance)
            return expert_instance
        except ImportError as e:
            raise ImportError(f"无法导入专家模块 {name}: {e}")
    
    def initialize_ocr_pipeline(self, mode: str = "full", debug: bool = True):
        """初始化OCR处理流水线
        
        Args:
            mode: 流水线模式
                - "full": 完整流水线（所有步骤）
                - "no_clustering": 不使用聚类步骤
                - "no_spelling": 不使用拼写纠错步骤
                - "no_compression": 不使用内容压缩步骤
                - "no_all": 不使用任何处理步骤
            debug: 是否输出调试信息
        """
        try:
            import sys
            algorithm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Algorithm")
            if algorithm_path not in sys.path:
                sys.path.insert(0, algorithm_path)
            
            from ocr_pipeline import create_pipeline
            self.ocr_pipeline = create_pipeline(mode=mode, debug=debug)
        except Exception as e:
            print(f"[ExpertManager] OCR流水线初始化失败: {e}")
            self.ocr_pipeline = None
    
    def process_with_experts(self, image, question: str, expert_names: List[str],
                            compression_ratio: float = 0.0, alpha: float = 0.6, beta: float = 0.4,
                            compression_method: str = "spatial",
                            use_pipeline: bool = False,
                            pipeline_mode: str = "full",
                            question_type: str = None,
                            debug: bool = False,
                            ocr_mode: str = "normal") -> Tuple[str, Dict[str, Any]]:
        """使用多个专家模块处理输入

        Args:
            image: 输入图像
            question: 问题文本
            expert_names: 专家模块名称列表
            compression_ratio: OCR prompt压缩率（已废弃，保留兼容性）
            alpha: 置信度权重（已废弃，保留兼容性）
            beta: TF-IDF权重（已废弃，保留兼容性）
            compression_method: 压缩方法（已废弃，保留兼容性）
            use_pipeline: 是否使用新的OCR流水线
            pipeline_mode: 流水线模式（full/no_clustering/no_spelling/no_compression/no_all）
            question_type: 数据集问题类型（如mydatavqa的question_type字段）
            debug: 是否输出调试信息
            ocr_mode: OCR处理模式（normal/random/LR-TB/direct）

        Returns:
            组合后的prompt
        """
        expert_outputs = []
        stats = {}  # 初始化stats

        for expert_name in expert_names:
            expert = self.get_expert(expert_name)
            if expert and expert.is_available():
                try:
                    result = expert.process(image, question)
                    
                    if use_pipeline and expert_name == "ocr":
                        # 使用OCR专家的流水线方法
                        if hasattr(expert, 'ocr_pipeline') and expert.ocr_pipeline is not None:
                            # 流水线已初始化，直接使用
                            ocr_texts = result.get("ocr_texts", [])
                            result_tuple = expert.ocr_pipeline.process_and_to_prompt(
                                ocr_texts, 
                                question=question,
                                dataset_type=question_type
                            )
                            # 处理返回值，可能是(answer, stats)或(prompt, stats)
                            if len(result_tuple) == 2:
                                prompt_text, stats = result_tuple
                            else:
                                # 二阶段推理模式：(answer, stats, full_prompt)
                                answer, stats, full_prompt = result_tuple
                                # 保存完整prompt到stats
                                stats["full_prompt"] = full_prompt
                                prompt_text = full_prompt
                        elif hasattr(expert, 'initialize_pipeline'):
                            # 初始化流水线
                            expert.initialize_pipeline(mode=pipeline_mode, debug=debug)
                            if expert.ocr_pipeline is not None:
                                ocr_texts = result.get("ocr_texts", [])
                                result_tuple = expert.ocr_pipeline.process_and_to_prompt(
                                    ocr_texts, 
                                    question=question,
                                    dataset_type=question_type
                                )
                                # 处理返回值，可能是(answer, stats)或(prompt, stats)
                                if len(result_tuple) == 2:
                                    prompt_text, stats = result_tuple
                                else:
                                    # 二阶段推理模式：(answer, stats, full_prompt)
                                    answer, stats, full_prompt = result_tuple
                                    # 保存完整prompt到stats
                                    stats["full_prompt"] = full_prompt
                                    prompt_text = full_prompt
                            else:
                                prompt_text = expert.to_prompt(result, mode=ocr_mode)
                        else:
                            prompt_text = expert.to_prompt(result, mode=ocr_mode)
                    else:
                        prompt_text = expert.to_prompt(result, mode=ocr_mode) if expert_name == "ocr" else expert.to_prompt(result)
                    
                    expert_outputs.append(prompt_text)
                except Exception as e:
                    print(f"专家模块 {expert_name} 处理失败: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 组合所有专家输出
        combined_prompt = "\n".join(expert_outputs)
        
        # 检查是否是二阶段推理模式（full_prompt已包含完整提示）
        if "full_prompt" in stats:
            final_prompt = combined_prompt
        else:
            # 非二阶段模式，添加标准提示
            final_prompt = f"{combined_prompt}\n\nBased on the above information, please answer the following question: {question}"
        
        # 返回prompt和stats
        return final_prompt, stats
    
    def get_available_experts(self) -> List[str]:
        """获取所有可用专家模块"""
        return list(self.experts.keys())