#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR算法流水线测试脚本
测试各个步骤是否正常工作，每步输出debug信息
"""

import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/hub"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Algorithm import (
    DBSCANClustering,
    SpellingCorrector,
    QuestionClassifier,
    ContentCompressor,
    OCRPipeline,
    create_pipeline
)


def test_dbscan_clustering():
    """测试DBSCAN聚类模块"""
    print("\n" + "="*60)
    print("测试 DBSCAN 聚类模块")
    print("="*60)
    
    mock_ocr = [
        {"text": "GeoCalib EPE Error", "polygon": [[10, 0], [200, 0], [200, 15], [10, 15]], "confidence": 0.95},
        {"text": "Cumulative Frequency", "polygon": [[10, 20], [200, 20], [200, 35], [10, 35]], "confidence": 0.92},
        {"text": "1.0", "polygon": [[5, 40], [30, 40], [30, 55], [5, 55]], "confidence": 0.98},
        {"text": "0.8", "polygon": [[5, 60], [30, 60], [30, 75], [5, 75]], "confidence": 0.97},
        {"text": "Method A", "polygon": [[300, 0], [400, 0], [400, 15], [300, 15]], "confidence": 0.94},
        {"text": "Method B", "polygon": [[300, 20], [400, 20], [400, 35], [300, 35]], "confidence": 0.93},
    ]
    
    clustering = DBSCANClustering(eps=50.0, debug=True)
    result = clustering.cluster(mock_ocr)
    
    print(f"\n聚类结果: {len(result)} 个块")
    for i, block in enumerate(result):
        print(f"  块{i+1}: {block['text'][:50]}...")
    
    return result


def test_spelling_correction():
    """测试拼写纠错模块"""
    print("\n" + "="*60)
    print("测试拼写纠错模块")
    print("="*60)
    
    mock_blocks = [
        {"text": "iamastudent", "minx": 0, "miny": 0, "maxx": 100, "maxy": 20},
        {"text": "thechartshowsthetrend", "minx": 0, "miny": 30, "maxx": 100, "maxy": 50},
        {"text": "MethodAhasbetterperformance", "minx": 0, "miny": 60, "maxx": 100, "maxy": 80},
        {"text": "normal text here", "minx": 0, "miny": 90, "maxx": 100, "maxy": 110},
    ]
    
    corrector = SpellingCorrector(debug=True)
    result = corrector.correct_blocks(mock_blocks)
    
    print(f"\n纠错结果:")
    for i, block in enumerate(result):
        orig = block.get("original_text", "")
        corr = block["text"]
        if orig != corr:
            print(f"  块{i+1}: '{orig}' -> '{corr}'")
        else:
            print(f"  块{i+1}: 无变化 '{corr}'")
    
    return result


def test_question_classifier():
    """测试问题分类模块"""
    print("\n" + "="*60)
    print("测试问题分类模块")
    print("="*60)
    
    test_questions = [
        "What is the trend of the data?",
        "Compare method A and B, which is better?",
        "Where is the title located?",
        "Describe the content of this chart.",
        "Calculate the difference between A and B.",
        "What is the total sum of values?",
    ]
    
    classifier = QuestionClassifier(debug=True)
    
    print("\n分类结果:")
    for question in test_questions:
        q_type, confidence = classifier.classify(question)
        print(f"  问题: '{question[:40]}...'")
        print(f"    -> 类型: {q_type}, 置信度: {confidence:.3f}")
    
    return classifier


def test_content_compressor():
    """测试内容压缩模块"""
    print("\n" + "="*60)
    print("测试内容压缩模块")
    print("="*60)
    
    mock_blocks = [
        {"text": "The trend shows increase from 0.1 to 0.9", "minx": 0, "miny": 0, "maxx": 100, "maxy": 20},
        {"text": "Method A has score 95.5", "minx": 0, "miny": 30, "maxx": 100, "maxy": 50},
        {"text": "This is a decorative text with no useful info", "minx": 0, "miny": 60, "maxx": 100, "maxy": 80},
        {"text": "Calculate total: 100 + 200 = 300", "minx": 0, "miny": 90, "maxx": 100, "maxy": 110},
    ]
    
    compressor = ContentCompressor(debug=True)
    
    test_cases = [
        ("trend", "What is the trend?"),
        ("comparison", "Compare A and B"),
        ("calculation", "Calculate the total"),
        ("comprehension", "Describe the chart"),
    ]
    
    for q_type, question in test_cases:
        print(f"\n--- 问题类型: {q_type} ---")
        result, stats = compressor.compress_blocks(mock_blocks, q_type, question)
        print(f"压缩统计: {stats}")
    
    return compressor


def test_full_pipeline():
    """测试完整流水线"""
    print("\n" + "="*60)
    print("测试完整OCR流水线")
    print("="*60)
    
    mock_ocr = [
        {"text": "GeoCalibEPEErrorCumulativeFrequency", "polygon": [[10, 0], [200, 0], [200, 15], [10, 15]], "confidence": 0.95},
        {"text": "WildCamera EPE ErrorCumulative Frequency", "polygon": [[10, 20], [200, 20], [200, 35], [10, 35]], "confidence": 0.92},
        {"text": "1.0", "polygon": [[5, 40], [30, 40], [30, 55], [5, 55]], "confidence": 0.98},
        {"text": "0.8", "polygon": [[5, 60], [30, 60], [30, 75], [5, 75]], "confidence": 0.97},
        {"text": "Method A score 95.5", "polygon": [[300, 0], [400, 0], [400, 15], [300, 15]], "confidence": 0.94},
        {"text": "Method B score 87.3", "polygon": [[300, 20], [400, 20], [400, 35], [300, 35]], "confidence": 0.93},
        {"text": "increase trend from 2020 to 2023", "polygon": [[300, 40], [500, 40], [500, 55], [300, 55]], "confidence": 0.91},
    ]
    
    test_cases = [
        ("What is the trend of the data?", "trend"),
        ("Compare Method A and B, which is better?", "comparison"),
        ("Calculate the difference between A and B scores", "calculation"),
        ("Describe the content of this chart", "comprehension"),
    ]
    
    for question, expected_type in test_cases:
        print(f"\n{'='*60}")
        print(f"测试问题: {question}")
        print(f"预期类型: {expected_type}")
        print("="*60)
        
        pipeline = create_pipeline(mode="full", debug=True)
        prompt, stats = pipeline.process_and_to_prompt(mock_ocr, question)
        
        print(f"\n最终Prompt (前200字符):")
        print(f"  {prompt[:200]}...")
        print(f"\n处理统计:")
        print(f"  问题类型: {stats.get('question_type', 'N/A')}")
        print(f"  最终块数: {stats.get('final_blocks', 0)}")
        print(f"  总耗时: {stats.get('total_time', 0):.3f}s")


def test_ablation_modes():
    """测试消融实验模式"""
    print("\n" + "="*60)
    print("测试消融实验模式")
    print("="*60)
    
    mock_ocr = [
        {"text": "TestBlock1 with numbers 123", "polygon": [[10, 0], [200, 0], [200, 15], [10, 15]], "confidence": 0.95},
        {"text": "TestBlock2 trend increase", "polygon": [[10, 20], [200, 20], [200, 35], [10, 35]], "confidence": 0.92},
    ]
    
    modes = ["full", "no_clustering", "no_spelling", "no_compression", "no_all"]
    
    for mode in modes:
        print(f"\n--- 模式: {mode} ---")
        pipeline = create_pipeline(mode=mode, debug=False)
        prompt, stats = pipeline.process_and_to_prompt(mock_ocr, "What is the trend?")
        print(f"  最终块数: {stats.get('final_blocks', 0)}")
        print(f"  Prompt长度: {len(prompt)}")


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# OCR算法流水线测试")
    print("#"*60)
    
    print("\n[1/6] 测试DBSCAN聚类模块...")
    test_dbscan_clustering()
    
    print("\n[2/6] 测试拼写纠错模块...")
    test_spelling_correction()
    
    print("\n[3/6] 测试问题分类模块...")
    test_question_classifier()
    
    print("\n[4/6] 测试内容压缩模块...")
    test_content_compressor()
    
    print("\n[5/6] 测试完整流水线...")
    test_full_pipeline()
    
    print("\n[6/6] 测试消融实验模式...")
    test_ablation_modes()
    
    print("\n" + "#"*60)
    print("# 所有测试完成!")
    print("#"*60)


if __name__ == "__main__":
    main()
