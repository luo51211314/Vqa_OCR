import sys
sys.path.append('/root/autodl-tmp/codes/Vqa_ocr')
import pandas as pd
import ast
from loaders.docvqa import Dataset

def parse_ground_truth(gt_str):
    """解析ground_truth字段，处理字符串格式的列表"""
    try:
        # 尝试解析为Python列表
        if isinstance(gt_str, str) and gt_str.startswith('['):
            return ast.literal_eval(gt_str)
        elif isinstance(gt_str, str):
            return [gt_str]
        elif isinstance(gt_str, list):
            return gt_str
        else:
            return [str(gt_str)]
    except:
        return [str(gt_str)]

def check_relaxed_accuracy_for_csv():
    """检查CSV文件中所有预测的relaxed_accuracy"""
    csv_path = "/root/autodl-tmp/codes/Vqa_ocr/results/docvqa_validation_relaxed_accuracy_detail.csv"
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"CSV文件共包含 {len(df)} 条数据")
    print("=" * 80)
    
    # 检查每一行的预测准确性
    correct_count = 0
    incorrect_count = 0
    
    results = []
    
    for idx, row in df.iterrows():
        question_id = row['question_id']
        question = row['question']
        predicted = str(row['predicted_answer'])
        ground_truth = parse_ground_truth(row['ground_truth'])
        
        # 使用metrics方法检查relaxed_accuracy
        score_result = Dataset.metrics([predicted], [ground_truth], metric_type="relaxed_accuracy")
        is_correct = bool(score_result["relaxed_accuracy"])
        
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
        
        results.append({
            'index': idx,
            'question_id': question_id,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'score': score_result["relaxed_accuracy"]
        })
    
    print(f"正确预测: {correct_count} 条")
    print(f"错误预测: {incorrect_count} 条")
    print(f"准确率: {correct_count/len(df)*100:.2f}%")
    print("=" * 80)
    
    # 交互式查看详细结果
    while True:
        print("\n选择操作:")
        print("1. 查看所有错误预测")
        print("2. 查看所有正确预测") 
        print("3. 按索引查看特定样本")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            # 查看错误预测
            incorrect_results = [r for r in results if not r['is_correct']]
            print(f"\n找到 {len(incorrect_results)} 条错误预测:")
            for i, result in enumerate(incorrect_results):
                print(f"{i+1}. ID: {result['question_id']}, 索引: {result['index']}")
                print(f"   预测: '{result['predicted']}'")
                print(f"   参考答案: {result['ground_truth']}")
                print(f"   分数: {result['score']}")
                print("-" * 60)
                
                if (i + 1) % 5 == 0:
                    cont = input("继续查看? (y/n): ").strip().lower()
                    if cont != 'y':
                        break
        
        elif choice == '2':
            # 查看正确预测
            correct_results = [r for r in results if r['is_correct']]
            print(f"\n找到 {len(correct_results)} 条正确预测:")
            for i, result in enumerate(correct_results):
                print(f"{i+1}. ID: {result['question_id']}, 索引: {result['index']}")
                print(f"   预测: '{result['predicted']}'")
                print(f"   参考答案: {result['ground_truth']}")
                print(f"   分数: {result['score']}")
                print("-" * 60)
                
                if (i + 1) % 5 == 0:
                    cont = input("继续查看? (y/n): ").strip().lower()
                    if cont != 'y':
                        break
        
        elif choice == '3':
            # 按索引查看
            try:
                index = int(input("请输入样本索引: ").strip())
                if 0 <= index < len(results):
                    result = results[index]
                    print(f"\n样本索引: {index}")
                    print(f"问题ID: {result['question_id']}")
                    print(f"问题: {df.iloc[index]['question'][:100]}...")
                    print(f"预测答案: '{result['predicted']}'")
                    print(f"参考答案: {result['ground_truth']}")
                    print(f"是否正确: {'是' if result['is_correct'] else '否'}")
                    print(f"分数: {result['score']}")
                    
                    # 显示详细的问题内容
                    show_full = input("显示完整问题内容? (y/n): ").strip().lower()
                    if show_full == 'y':
                        print(f"完整问题:\n{df.iloc[index]['question']}")
                else:
                    print("索引超出范围")
            except ValueError:
                print("请输入有效的数字")
        
        elif choice == '4':
            break
        
        else:
            print("无效选择")

if __name__ == "__main__":
    check_relaxed_accuracy_for_csv()

# 测试用例
test_cases = [
    # (预测答案, 参考答案, 期望结果)
    ("hello world", "hello", True),      # 包含所有字符
    ("hello", "hello world", False),     # 不包含所有字符  
    ("HELLO", "hello", True),           # 大小写不敏感
    ("hello123", "hello", True),         # 包含所有字符
    ("helo", "hello", False),           # 缺少字符'l'
    ("", "hello", False),               # 空预测
    ("hello", "", True),                # 空参考答案
]

print("测试relaxed_accuracy逻辑:")
print("=" * 50)

for i, (pred, ref, expected) in enumerate(test_cases, 1):
    # 使用实际的metrics方法
    result = Dataset._metrics_relaxed_accuracy([pred], [[ref]])
    actual_score = result["relaxed_accuracy"]
    actual_bool = bool(actual_score)
    
    # 直接测试all逻辑
    pred_lower = str(pred).strip().lower()
    ref_lower = str(ref).strip().lower()
    direct_test = all(char in pred_lower for char in ref_lower) if ref_lower else True
    
    status = "✓" if actual_bool == expected else "✗"
    
    print(f"测试 {i}: {status}")
    print(f"  预测: '{pred}' -> '{pred_lower}'")
    print(f"  参考: '{ref}' -> '{ref_lower}'")
    print(f"  期望: {expected}, 实际: {actual_bool}, 分数: {actual_score}")
    print(f"  直接测试: {direct_test}")
    print("-" * 30)

# 运行测试
print("\n运行测试脚本:")
print("=" * 50)