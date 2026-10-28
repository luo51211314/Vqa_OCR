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
        # φ（OCR prompt 序列化）的全局配置，由主程序在解析参数后注入
        self.ocr_config: Dict[str, Any] = {
            "order": "native",        # native/random/ros/coordinate
            "template": "p2",         # p1/p2/p3
            "budget_tokens": None,    # None/0 = 不限
            "conf_threshold": 0.0,    # L2 置信度过滤（主表关闭）
            "dedup": False,           # L2 去重（主表关闭）
        }
        self.token_counter = None
        # 逐样本 φ 统计（token 增量/丢弃块数），供 metrics.json 汇总
        self.phi_stats: List[Dict[str, Any]] = []

    def set_ocr_config(self, **kwargs):
        """更新 φ 配置（order/template/budget_tokens/conf_threshold/dedup）。"""
        for key, value in kwargs.items():
            self.ocr_config[key] = value

    def set_token_counter(self, token_counter):
        """注入目标模型 tokenizer 计数函数 fn(str)->int（token 预算消融用）。"""
        self.token_counter = token_counter

    def get_ocr_summary(self) -> Dict[str, Any]:
        """汇总本次评测所有样本的 φ 统计（无 OCR 样本时返回空 dict）。"""
        if not self.phi_stats:
            return {}
        n = len(self.phi_stats)
        def _mean(key):
            vals = [s.get(key) for s in self.phi_stats if s.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None
        def _l2_totals(name):
            kept = dropped = missing = 0
            observed = False
            for sample in self.phi_stats:
                stats = ((sample.get("l2_filters") or {}).get(name) or {})
                if not stats:
                    continue
                observed = True
                kept += int(stats.get("kept", 0))
                dropped += int(stats.get("dropped", 0))
                missing += int(stats.get("missing_confidence", 0))
            total = kept + dropped
            return {
                "kept": kept, "dropped": dropped, "missing": missing,
                "retention_rate": round(kept / total, 4) if total else None,
                "drop_rate": round(dropped / total, 4) if total else None,
                "missing_rate": round(missing / total, 4) if total else None,
            } if observed else {}
        dropped = sum(s.get("n_blocks_dropped", 0) for s in self.phi_stats)
        blocks = sum(s.get("n_blocks_in", 0) for s in self.phi_stats)
        conf_stats = _l2_totals("confidence_filter")
        dedup_stats = _l2_totals("dedup")
        return {
            "n_samples": n,
            "order": self.phi_stats[0].get("order"),
            "template": self.phi_stats[0].get("template"),
            "mean_total_tokens": _mean("total_tokens"),
            "mean_ocr_tokens": _mean("ocr_tokens"),
            "mean_question_tokens": _mean("question_tokens"),
            "mean_delta_tokens": _mean("delta_tokens"),
            "blocks_total": blocks,
            "blocks_dropped_by_budget": dropped,
            "block_drop_rate": round(dropped / blocks, 4) if blocks else 0.0,
            "blocks_kept_by_confidence": conf_stats.get("kept"),
            "blocks_dropped_by_confidence": conf_stats.get("dropped"),
            "confidence_retention_rate": conf_stats.get("retention_rate"),
            "missing_confidence_rate": conf_stats.get("missing_rate"),
            "blocks_kept_after_dedup": dedup_stats.get("kept"),
            "blocks_dropped_by_dedup": dedup_stats.get("dropped"),
            "dedup_retention_rate": dedup_stats.get("retention_rate"),
            "dedup_drop_rate": dedup_stats.get("drop_rate"),
        }
    
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
        ocr_self_contained = False  # OCR 输出是否已在 φ 模板内包含问题

        # OCR 专家最后渲染，保证非 OCR 专家（text/chart）输出在其上方
        ordered_names = [n for n in expert_names if n != "ocr"] + (
            ["ocr"] if "ocr" in expert_names else []
        )

        for expert_name in ordered_names:
            expert = self.get_expert(expert_name)
            if not (expert and expert.is_available()):
                continue
            try:
                result = expert.process(image, question)

                if expert_name != "ocr":
                    expert_outputs.append(expert.to_prompt(result))
                    continue

                # ---- OCR 专家 ----
                if use_pipeline:
                    prompt_text, stats = self._run_ocr_pipeline(
                        expert, result, question, question_type, pipeline_mode, debug
                    )
                    # pipeline 输出（含二阶段 full_prompt）均已自带问题
                    ocr_self_contained = True
                else:
                    # L2 有损过滤（主表配置下等价不过滤）
                    l2_blocks, l2_stats = self._apply_l2(result.get("ocr_texts", []))
                    result_for_prompt = dict(result)
                    result_for_prompt["ocr_texts"] = l2_blocks

                    effective_order = ocr_mode or self.ocr_config["order"]
                    prompt_text, phi_stats = expert.to_prompt(
                        result_for_prompt,
                        mode=effective_order,
                        question=question,
                        template=self.ocr_config["template"],
                        budget_tokens=self.ocr_config["budget_tokens"],
                        token_counter=self.token_counter,
                        return_stats=True,
                    )
                    if phi_stats:
                        phi_stats["l2_filters"] = l2_stats
                        self.phi_stats.append(phi_stats)
                    stats = {"l2_filters": l2_stats}
                    ocr_self_contained = True

                expert_outputs.append(prompt_text)
            except Exception as e:
                print(f"专家模块 {expert_name} 处理失败: {e}")
                import traceback
                traceback.print_exc()

        combined_prompt = "\n".join(expert_outputs)

        if ocr_self_contained:
            # φ 模板内已含问题，禁止再追加一层旧包装（旧实现会重复拼接问题）
            final_prompt = combined_prompt
        else:
            final_prompt = f"{combined_prompt}\n\nBased on the above information, please answer the following question: {question}"

        return final_prompt, stats

    def _apply_l2(self, ocr_texts):
        """应用 L2 有损过滤（置信度/去重），主表配置下为恒等操作。"""
        try:
            from Algorithm.ocr_filters import apply_l2_filters
        except ImportError:
            import sys
            algorithm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Algorithm")
            if algorithm_path not in sys.path:
                sys.path.insert(0, algorithm_path)
            from ocr_filters import apply_l2_filters
        return apply_l2_filters(
            ocr_texts,
            conf_threshold=self.ocr_config.get("conf_threshold", 0.0),
            dedup=self.ocr_config.get("dedup", False),
        )

    def _run_ocr_pipeline(self, expert, result, question, question_type, pipeline_mode, debug):
        """调用 Algorithm OCR 流水线（含惰性初始化），返回 (prompt, stats)。"""
        if not (hasattr(expert, "ocr_pipeline") and expert.ocr_pipeline is not None) \
                and hasattr(expert, "initialize_pipeline"):
            expert.initialize_pipeline(mode=pipeline_mode, debug=debug)

        if not (hasattr(expert, "ocr_pipeline") and expert.ocr_pipeline is not None):
            effective_order = self.ocr_config["order"]
            return expert.to_prompt(
                result, mode=effective_order, question=question,
                template=self.ocr_config["template"],
                budget_tokens=self.ocr_config["budget_tokens"],
                token_counter=self.token_counter,
                return_stats=True,
            )

        ocr_texts = result.get("ocr_texts", [])
        result_tuple = expert.ocr_pipeline.process_and_to_prompt(
            ocr_texts, question=question, dataset_type=question_type
        )
        if len(result_tuple) == 2:
            return result_tuple[0], result_tuple[1]
        # 二阶段推理：(answer, stats, full_prompt)
        _answer, p_stats, full_prompt = result_tuple
        p_stats["full_prompt"] = full_prompt
        return full_prompt, p_stats
    
    def get_available_experts(self) -> List[str]:
        """获取所有可用专家模块"""
        return list(self.experts.keys())