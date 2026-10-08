import os
import numpy as np
from typing import List, Dict, Any, Tuple
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"


class QuestionClassifier:
    """BERT问题分类模块
    
    使用轻量BERT（bert-tiny）对问题文本进行分类，归到5类问题类型：
    - trend: 趋势类问题
    - comparison: 对比类问题
    - location: 位置类问题
    - comprehension: 理解类问题
    - calculation: 计算类问题
    """
    
    REF_SENTENCES = {
        "trend": "What is the trend of the data? Is it increasing or decreasing?",
        "comparison": "Compare the performance of method A, B and C. Which is better?",
        "location": "Where is the text located? Is it on the top or left?",
        "comprehension": "Describe the content of the image. What is the main idea?",
        "calculation": "Calculate the difference between A and B. What is the total sum?"
    }
    
    FEATURE_WORDS = {
        "trend": {"trend", "increase", "decrease", "rise", "drop", "up", "down", "grow", "change", "from", "to"},
        "comparison": {"compare", "better", "worse", "higher", "lower", "than", "best", "worst", "method", "versus", "vs"},
        "location": {"where", "position", "top", "bottom", "left", "right", "center", "upper", "lower", "which section", "locate", "area", "region"},
        "comprehension": {"describe", "what is", "content", "main idea", "title", "explain", "what does", "mean", "summary", "about"},
        "calculation": {"calculate", "difference", "sum", "total", "minus", "plus", "average", "how much", "score of", "subtract", "add", "count", "how many"}
    }
    
    def __init__(self, model_name: str = "prajjwal1/bert-tiny", debug: bool = True, preloaded_model: Dict[str, Any] = None):
        """
        Args:
            model_name: BERT模型名称，默认使用bert-tiny（超轻量）
            debug: 是否输出调试信息
            preloaded_model: 预加载的模型字典，包含tokenizer, model, device
        """
        self.model_name = model_name
        self.debug = debug
        self.model = None
        self.tokenizer = None
        self.ref_embeddings = None
        self._initialized = False
        
        # 如果提供了预加载模型，直接使用
        if preloaded_model:
            self.tokenizer = preloaded_model.get("tokenizer")
            self.model = preloaded_model.get("model")
            self.device = preloaded_model.get("device", "cpu")
            if self.model and self.tokenizer:
                self.ref_embeddings = self._compute_reference_embeddings()
                self._initialized = True
                if self.debug:
                    print(f"[问题分类] 使用预加载的BERT模型，设备: {self.device}")
    
    def _check_model_cached(self, model_name):
        """检查模型是否已在本地缓存"""
        try:
            from transformers.utils.hub import cached_file
            cached_path = cached_file(model_name, "config.json", _raise_exceptions_for_missing_entries=False)
            return cached_path is not None
        except:
            return False
    
    def _lazy_init(self):
        """延迟初始化模型"""
        if self._initialized:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
            from transformers.utils.hub import cached_file
            import torch
            
            if self.debug:
                print(f"[问题分类] 加载BERT模型: {self.model_name}")
            
            # 检查模型是否已缓存
            model_cached = self._check_model_cached(self.model_name)
            if model_cached:
                print(f"[问题分类] 模型已缓存，使用离线模式加载")
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                print(f"[问题分类] 模型未缓存，允许在线下载")
                os.environ.pop("HF_HUB_OFFLINE", None)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.device = device
            
            # 恢复离线模式设置
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            self.ref_embeddings = self._compute_reference_embeddings()
            
            self._initialized = True
            if self.debug:
                print(f"[问题分类] 模型加载成功，设备: {device}")
                
        except Exception as e:
            print(f"[问题分类] BERT模型加载失败: {e}")
            print("[问题分类] 将使用关键词匹配方法进行分类")
            os.environ["HF_HUB_OFFLINE"] = "1"  # 确保恢复离线模式
            self.model = None
            self.tokenizer = None
            self._initialized = True
    
    def _compute_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """计算参考句的嵌入向量"""
        import torch
        
        embeddings = {}
        
        with torch.no_grad():
            for q_type, ref_text in self.REF_SENTENCES.items():
                inputs = self.tokenizer(
                    ref_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                embeddings[q_type] = embedding
        
        return embeddings
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        import torch
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def _keyword_based_classify(self, question: str) -> Tuple[str, float]:
        """基于关键词的问题分类（作为后备方案）
        
        Args:
            question: 问题文本
            
        Returns:
            (问题类型, 置信度)
        """
        question_lower = question.lower()
        
        scores = {}
        for q_type, keywords in self.FEATURE_WORDS.items():
            score = sum(1 for kw in keywords if kw in question_lower)
            scores[q_type] = score
        
        max_score = max(scores.values())
        if max_score == 0:
            return "comprehension", 0.3
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / (sum(scores.values()) + 1e-8)
        
        return best_type, min(confidence, 1.0)
    
    def classify(self, question: str) -> Tuple[str, float]:
        """对问题进行分类
        
        Args:
            question: 问题文本
            
        Returns:
            (问题类型, 置信度)
        """
        if not question or len(question.strip()) == 0:
            return "comprehension", 0.5
        
        self._lazy_init()
        
        # 先尝试关键词匹配，如果置信度高则直接返回
        keyword_type, keyword_confidence = self._keyword_based_classify(question)
        if keyword_confidence > 0.5:
            if self.debug:
                print(f"[问题分类] 关键词匹配置信度高({keyword_confidence:.2f})，使用关键词结果: {keyword_type}")
            return keyword_type, keyword_confidence
        
        # 否则尝试BERT分类
        if self.model is None or self.tokenizer is None:
            return keyword_type, keyword_confidence
        
        try:
            question_embedding = self._get_embedding(question)
            
            similarities = {}
            for q_type, ref_embedding in self.ref_embeddings.items():
                sim = self._cosine_similarity(question_embedding, ref_embedding)
                similarities[q_type] = sim
            
            best_type = max(similarities, key=similarities.get)
            confidence = similarities[best_type]
            
            # 如果BERT分类的置信度太低，使用关键词匹配结果
            if confidence < 0.6:
                if self.debug:
                    print(f"[问题分类] BERT分类置信度低({confidence:.2f})，使用关键词结果: {keyword_type}")
                return keyword_type, keyword_confidence
            
            return best_type, confidence
            
        except Exception as e:
            if self.debug:
                print(f"[问题分类] BERT分类失败: {e}，使用关键词方法")
            return keyword_type, keyword_confidence
    
    def classify_with_dataset_type(self, question: str, dataset_type: str = None) -> Tuple[str, float]:
        """结合数据集类型进行问题分类
        
        Args:
            question: 问题文本
            dataset_type: 数据集类型（如mydatavqa的问题类型字段）
            
        Returns:
            (问题类型, 置信度)
        """
        if dataset_type:
            dataset_type_lower = dataset_type.lower()
            
            type_mapping = {
                "trend": "trend",
                "comparison": "comparison",
                "location": "location",
                "comprehension": "comprehension",
                "calculation": "calculation",
                "numerical": "calculation",
                "numerical_calculation": "calculation",
            }
            
            for key, mapped_type in type_mapping.items():
                if key in dataset_type_lower:
                    return mapped_type, 1.0
        
        return self.classify(question)
    
    def batch_classify(self, questions: List[str]) -> List[Tuple[str, float]]:
        """批量分类问题
        
        Args:
            questions: 问题文本列表
            
        Returns:
            [(问题类型, 置信度), ...]
        """
        results = []
        for question in questions:
            q_type, confidence = self.classify(question)
            results.append((q_type, confidence))
        return results
    
    def get_question_type_info(self, question: str) -> Dict[str, Any]:
        """获取问题类型的详细信息
        
        Args:
            question: 问题文本
            
        Returns:
            包含类型、置信度、特征词等信息的字典
        """
        q_type, confidence = self.classify(question)
        
        question_lower = question.lower()
        matched_keywords = []
        if q_type in self.FEATURE_WORDS:
            matched_keywords = [kw for kw in self.FEATURE_WORDS[q_type] if kw in question_lower]
        
        return {
            "question": question,
            "type": q_type,
            "confidence": confidence,
            "matched_keywords": matched_keywords,
            "reference_sentence": self.REF_SENTENCES.get(q_type, "")
        }
