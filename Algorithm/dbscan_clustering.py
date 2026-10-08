import os
import sys
from typing import List, Dict, Any
import numpy as np
import hdbscan

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DBSCANClustering:
    """基于HDBSCAN的文本块聚类
    
    实现三步聚类：
    1. 按y坐标聚类（分行）
    2. 按x坐标聚类（分列）
    3. 选择行聚类或列聚类作为最终结果
    
    包含词性分析功能
    """
    
    def __init__(self, eps: float = 30.0, debug: bool = True):
        """
        Args:
            eps: DBSCAN的邻域半径
            debug: 是否输出调试信息
        """
        self.eps = eps
        self.debug = debug
        self.spacy_nlp = None
        self._init_spacy()
    
    def _init_spacy(self):
        """初始化Spacy用于词性分析"""
        try:
            import spacy
            self.spacy_nlp = spacy.load("en_core_web_sm")
            if self.debug:
                print("[聚类] Spacy初始化成功，词性分析已启用")
        except Exception as e:
            if self.debug:
                print(f"[聚类] Spacy初始化失败: {e}")
                print("[聚类] 词性分析功能不可用")
            self.spacy_nlp = None
    
    def _analyze_pos(self, text: str) -> Dict[str, Any]:
        """使用Spacy分析词性
        
        Args:
            text: 文本
            
        Returns:
            词性分析结果
        """
        if not self.spacy_nlp or not text:
            return {
                "nouns": [],
                "verbs": [],
                "numbers": [],
                "has_noun": False,
                "has_number": False,
                "block_type": "unknown",
                "pos_sequence": []
            }
        
        doc = self.spacy_nlp(text)
        
        nouns = []
        verbs = []
        numbers = []
        pos_sequence = []
        
        for token in doc:
            pos_tag = token.pos_
            pos_sequence.append((token.text, pos_tag))
            if pos_tag in ["NOUN", "PROPN"]:
                nouns.append(token.text)
            elif pos_tag in ["VERB", "AUX"]:
                verbs.append(token.text)
            elif pos_tag == "NUM":
                numbers.append(token.text)
        
        # 判断文本块类型
        block_type = self._classify_block_type(doc, pos_sequence)
        
        return {
            "nouns": nouns,
            "verbs": verbs,
            "numbers": numbers,
            "has_noun": len(nouns) > 0,
            "has_number": len(numbers) > 0,
            "block_type": block_type,
            "pos_sequence": pos_sequence
        }
    
    def _classify_block_type(self, doc, pos_sequence: List[tuple]) -> str:
        """分类文本块类型
        
        Args:
            doc: Spacy文档对象
            pos_sequence: 词性序列
            
        Returns:
            文本块类型：'word', 'noun_phrase', 'sentence', 'unknown'
        """
        if not pos_sequence:
            return "unknown"
        
        # 如果只有一个词
        if len(pos_sequence) == 1:
            return "word"
        
        # 检查是否所有词都是名词（名词短语）
        all_nouns = all(pos in ["NOUN", "PROPN"] for _, pos in pos_sequence)
        if all_nouns:
            return "noun_phrase"
        
        # 检查是否有介词（ADP）
        has_preposition = any(pos == "ADP" for _, pos in pos_sequence)
        has_noun = any(pos in ["NOUN", "PROPN"] for _, pos in pos_sequence)
        
        # 名词+介词混合出现，标记为句子
        if has_preposition and has_noun:
            return "sentence"
        
        # 其他情况标记为句子
        return "sentence"
    
    def _extract_block_coords(self, ocr_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从OCR结果中提取文本块坐标和文本
        
        Args:
            ocr_texts: OCR识别结果列表
            
        Returns:
            文本块列表，每个块包含文本和坐标信息
        """
        blocks = []
        
        for i, item in enumerate(ocr_texts):
            text = item.get('text', '')
            if not text.strip():
                continue
            
            # 计算边界框
            if 'polygon' in item:
                polygon = np.array(item['polygon'])
                minx = float(np.min(polygon[:, 0]))
                miny = float(np.min(polygon[:, 1]))
                maxx = float(np.max(polygon[:, 0]))
                maxy = float(np.max(polygon[:, 1]))
            elif 'coordinates' in item:
                coords = item['coordinates']
                minx = float(coords['minx'])
                miny = float(coords['miny'])
                maxx = float(coords['maxx'])
                maxy = float(coords['maxy'])
            else:
                # 如果没有坐标信息，使用默认值
                minx, miny, maxx, maxy = 0.0, 0.0, 100.0, 30.0
            
            # 计算中心点
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            
            # 计算宽度和高度
            width = maxx - minx
            height = maxy - miny
            
            # 计算文本长度
            text_length = len(text)
            
            # 判断是否为数值
            is_numeric = bool(self._is_numeric(text))
            
            block = {
                'text': text,
                'minx': minx,
                'miny': miny,
                'maxx': maxx,
                'maxy': maxy,
                'cx': cx,
                'cy': cy,
                'width': width,
                'height': height,
                'text_length': text_length,
                'is_numeric': is_numeric,
                'pos_info': self._analyze_pos(text)
            }
            
            blocks.append(block)
        
        return blocks
    
    def _is_numeric(self, text: str) -> bool:
        """判断文本是否为数值"""
        try:
            # 尝试转换为浮点数
            float(text.replace(',', '').replace('%', ''))
            return True
        except:
            return False
    
    def _normalize_labels(self, labels: np.ndarray) -> np.ndarray:
        """归一化标签，将-1（噪声）保持不变，其他标签从0开始重新编号"""
        if len(labels) == 0:
            return labels
        
        unique_labels = sorted(set(labels) - {-1})
        label_map = {label: i for i, label in enumerate(unique_labels)}
        normalized = np.array([label_map.get(l, -1) for l in labels])
        return normalized
    
    def _cluster_by_y(self, blocks: List[Dict[str, Any]]) -> np.ndarray:
        """第一步：按y坐标聚类，分行"""
        if len(blocks) <= 1:
            return np.array([0] * len(blocks))
        
        y_features = np.array([[b["cy"]] for b in blocks])
        
        clustering = hdbscan.HDBSCAN(
            min_samples=1,
            min_cluster_size=2,
            metric="manhattan",
            allow_single_cluster=True,
            cluster_selection_epsilon=10.0
        )
        labels = clustering.fit_predict(y_features)
        
        if self.debug:
            unique_labels = sorted(set(labels))
            n_rows = len(unique_labels) - (1 if -1 in unique_labels else 0)
            print(f"[图表聚类] 第一步-行聚类: 发现 {n_rows} 行")
            print(f"[图表聚类] 行标签分布:")
            for label in unique_labels:
                if label == -1:
                    continue
                indices = [i for i, l in enumerate(labels) if l == label]
                texts = [blocks[i]["text"][:20] for i in indices[:5]]
                y_coords = [blocks[i]["cy"] for i in indices]
                avg_y = sum(y_coords) / len(y_coords) if y_coords else 0
                print(f"  行{label} (y≈{avg_y:.1f}): {texts}")
        
        return labels
    
    def _cluster_by_x(self, blocks: List[Dict[str, Any]]) -> np.ndarray:
        """第二步：按x坐标聚类，分列"""
        if len(blocks) <= 1:
            return np.array([0] * len(blocks))
        
        x_features = np.array([[b["cx"]] for b in blocks])
        
        clustering = hdbscan.HDBSCAN(
            min_samples=1,
            min_cluster_size=2,
            metric="manhattan",
            allow_single_cluster=True,
            cluster_selection_epsilon=10.0
        )
        labels = clustering.fit_predict(x_features)
        
        if self.debug:
            unique_labels = sorted(set(labels))
            n_cols = len(unique_labels) - (1 if -1 in unique_labels else 0)
            print(f"[图表聚类] 第二步-列聚类: 发现 {n_cols} 列")
            print(f"[图表聚类] 列标签分布:")
            for label in unique_labels:
                if label == -1:
                    continue
                indices = [i for i, l in enumerate(labels) if l == label]
                texts = [blocks[i]["text"][:20] for i in indices[:5]]
                x_coords = [blocks[i]["cx"] for i in indices]
                avg_x = sum(x_coords) / len(x_coords) if x_coords else 0
                print(f"  列{label} (x≈{avg_x:.1f}): {texts}")
        
        return labels
    
    def _analyze_cluster_structure(self, blocks: List[Dict[str, Any]], labels: np.ndarray, is_row_cluster: bool) -> Dict[str, Any]:
        """分析聚类结构，判断是否符合图表元数据模式
        
        图表元数据模式：
        1. <名词 数值, 数值, ...> - 一个名词对应多个数值，顺序正确
        2. <专有名词/名词短语 句子> - 层次结构
        
        Args:
            blocks: 文本块列表
            labels: 聚类标签
            is_row_cluster: 是否为行聚类
            
        Returns:
            分析结果
        """
        unique_labels = sorted(set(labels) - {-1})
        valid_clusters = []
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_blocks = [blocks[i] for i in cluster_indices]
            
            if len(cluster_blocks) < 2:
                continue
            
            # 分析聚类中的文本类型
            noun_count = 0
            number_count = 0
            noun_phrase_count = 0
            sentence_count = 0
            
            for block in cluster_blocks:
                pos_info = block.get('pos_info', {})
                block_type = pos_info.get('block_type', 'unknown')
                
                if block['is_numeric']:
                    number_count += 1
                elif block_type == 'word':
                    # 单个词，检查词性
                    if pos_info.get('has_noun', False):
                        noun_count += 1
                elif block_type == 'noun_phrase':
                    noun_phrase_count += 1
                elif block_type == 'sentence':
                    sentence_count += 1
            
            # 检查是否符合图表元数据模式
            is_metadata_pattern = False
            pattern_type = ""
            
            # 模式1: <名词 数值, 数值, ...> - 顺序正确，名词在前
            if (noun_count + noun_phrase_count) == 1 and number_count >= 1:
                is_metadata_pattern = True
                pattern_type = "noun_numbers"
            # 模式2: <专有名词/名词短语 句子> - 层次结构
            elif (noun_count + noun_phrase_count) >= 1 and sentence_count >= 1:
                is_metadata_pattern = True
                pattern_type = "noun_sentence"
            # 模式3: 多个名词短语（层次结构）
            elif noun_phrase_count >= 2 and number_count == 0:
                is_metadata_pattern = True
                pattern_type = "noun_phrases"
            
            valid_clusters.append({
                'label': label,
                'size': len(cluster_blocks),
                'noun_count': noun_count,
                'noun_phrase_count': noun_phrase_count,
                'sentence_count': sentence_count,
                'number_count': number_count,
                'is_metadata_pattern': is_metadata_pattern,
                'pattern_type': pattern_type,
                'blocks': cluster_blocks
            })
        
        metadata_clusters = [c for c in valid_clusters if c['is_metadata_pattern']]
        
        return {
            'total_clusters': len(unique_labels),
            'valid_clusters': len(valid_clusters),
            'metadata_clusters': len(metadata_clusters),
            'metadata_ratio': len(metadata_clusters) / len(valid_clusters) if valid_clusters else 0,
            'clusters': valid_clusters
        }
    
    def _select_best_clustering(self, blocks: List[Dict[str, Any]], row_labels: np.ndarray, col_labels: np.ndarray) -> str:
        """选择行聚类或列聚类作为最终结果
        
        选择标准：
        1. 哪个聚类的元数据模式匹配数量更多
        2. 如果数量相同，默认选择行聚类
        
        Args:
            blocks: 文本块列表
            row_labels: 行聚类标签
            col_labels: 列聚类标签
            
        Returns:
            'row' 或 'col'
        """
        # 分析行聚类
        row_analysis = self._analyze_cluster_structure(blocks, row_labels, is_row_cluster=True)
        # 分析列聚类
        col_analysis = self._analyze_cluster_structure(blocks, col_labels, is_row_cluster=False)
        
        if self.debug:
            print(f"[图表聚类] 行聚类分析: 元数据模式 {row_analysis['metadata_clusters']}/{row_analysis['valid_clusters']}")
            print(f"[图表聚类] 列聚类分析: 元数据模式 {col_analysis['metadata_clusters']}/{col_analysis['valid_clusters']}")
        
        # 比较元数据模式数量
        if row_analysis['metadata_clusters'] > col_analysis['metadata_clusters']:
            return 'row'
        elif col_analysis['metadata_clusters'] > row_analysis['metadata_clusters']:
            return 'col'
        
        # 如果元数据模式数量相同，默认选择行聚类
        if self.debug:
            print(f"[图表聚类] 两种聚类方法元数据模式数量相同，默认选择行聚类")
        return 'row'
    
    def _merge_blocks_by_label(self, blocks: List[Dict[str, Any]], labels: np.ndarray, label_type: str) -> List[Dict[str, Any]]:
        """根据标签合并文本块
        
        Args:
            blocks: 文本块列表
            labels: 聚类标签
            label_type: 'row' 或 'col'
            
        Returns:
            合并后的块列表
        """
        unique_labels = sorted(set(labels) - {-1})
        merged_blocks = []
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_blocks = [blocks[i] for i in cluster_indices]
            
            # 按坐标排序
            if label_type == 'row':
                # 行聚类：按x坐标排序
                cluster_blocks.sort(key=lambda b: b["cx"])
            else:
                # 列聚类：按y坐标排序
                cluster_blocks.sort(key=lambda b: b["cy"])
            
            # 计算合并后的边界框
            minx = min(b["minx"] for b in cluster_blocks)
            miny = min(b["miny"] for b in cluster_blocks)
            maxx = max(b["maxx"] for b in cluster_blocks)
            maxy = max(b["maxy"] for b in cluster_blocks)
            
            # 合并文本
            merged_texts = [b["text"] for b in cluster_blocks]
            merged_text = f"text: {' '.join(merged_texts)}"
            
            # 对合并后的文本进行词性分析
            merged_pos_info = self._analyze_pos(" ".join(merged_texts))
            
            merged_block = {
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy,
                "text": merged_text,
                "merged_texts": merged_texts,
                "cluster_id": int(label),
                "is_noise": False,
                "label_type": label_type,
                "pos_info": merged_pos_info
            }
            
            merged_blocks.append(merged_block)
        
        # 按坐标排序最终结果
        if label_type == 'row':
            # 行聚类：按y坐标排序
            merged_blocks.sort(key=lambda b: (b["miny"], b["minx"]))
        else:
            # 列聚类：按x坐标排序
            merged_blocks.sort(key=lambda b: (b["minx"], b["miny"]))
        
        return merged_blocks
    
    def cluster(self, ocr_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行聚类
        
        Args:
            ocr_texts: OCR识别结果列表
            
        Returns:
            聚类后的块列表，每个块包含合并后的文本和坐标
        """
        if not ocr_texts:
            return []
        
        if len(ocr_texts) <= 1:
            return self._extract_block_coords(ocr_texts)
        
        blocks = self._extract_block_coords(ocr_texts)
        
        if self.debug:
            print(f"\n[图表聚类] 原始OCR文本块数量: {len(blocks)}")
            for i, block in enumerate(blocks[:5]):
                print(f"  块{i} [{block['cx']:.1f}, {block['cy']:.1f}]: {block['text'][:30]}")
            if len(blocks) > 5:
                print(f"  ... 还有 {len(blocks) - 5} 个块")
        
        # 第一步：行聚类
        row_labels = self._cluster_by_y(blocks)
        # 第二步：列聚类
        col_labels = self._cluster_by_x(blocks)
        # 第三步：选择最佳聚类
        best_cluster_type = self._select_best_clustering(blocks, row_labels, col_labels)
        
        if self.debug:
            print(f"[图表聚类] 选择最佳聚类类型: {best_cluster_type}")
        
        # 根据选择的聚类类型合并块
        if best_cluster_type == 'row':
            merged_blocks = self._merge_blocks_by_label(blocks, row_labels, 'row')
        else:
            merged_blocks = self._merge_blocks_by_label(blocks, col_labels, 'col')
        
        if self.debug:
            print(f"[图表聚类] 最终聚类结果: {len(merged_blocks)} 个块")
            for i, block in enumerate(merged_blocks[:5]):
                print(f"  聚类块{i} [{block['minx']:.1f}, {block['miny']:.1f}] - [{block['maxx']:.1f}, {block['maxy']:.1f}]: {block['text'][:50]}")
            if len(merged_blocks) > 5:
                print(f"  ... 还有 {len(merged_blocks) - 5} 个聚类块")
        
        return merged_blocks