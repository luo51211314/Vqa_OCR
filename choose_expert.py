"""
专家模块选择器 - 根据数据集类型自动选择最合适的专家模块
"""
from typing import List, Dict

class ExpertChooser:
    """专家选择器类"""
    
    # 数据集到专家模块的映射
    DATASET_TO_EXPERTS = {
        # 文档理解任务
        "docvqa": ["text"],
        "funsd": ["text"],
        "rvlcdip": ["text"],
        
        # 图表分析任务  
        "chartqa": ["chart"],
        "plotqa": ["chart"],
        "dvqa": ["chart"],
        
        # 科学问答（可能需要多种专家）
        "scienceqa": ["text", "chart"],
        
        # 通用视觉问答
        "gqa": [],  # 不使用专家模块
        "vqa": []   # 不使用专家模块
    }
    
    # 专家模块的默认模型路径配置
    EXPERT_MODEL_PATHS = {
        "text": {
            "detector": "/root/autodl-tmp/models/dbnetpp",
            "recognizer": "/root/autodl-tmp/models/svtr"
        },
        "chart": {
            "model": "/root/autodl-tmp/model/pix2struct_hug"
        },
        "icon": {
            "model": "/root/autodl-tmp/models/icon_classifier"
        }
    }
    
    @classmethod
    def choose_experts_for_dataset(cls, dataset_name: str) -> List[str]:
        """根据数据集名称选择专家模块"""
        return cls.DATASET_TO_EXPERTS.get(dataset_name.lower(), [])
    
    @classmethod
    def get_expert_config(cls, expert_name: str) -> Dict:
        """获取专家模块的配置"""
        return cls.EXPERT_MODEL_PATHS.get(expert_name, {})
    
    @classmethod
    def should_use_experts(cls, dataset_name: str) -> bool:
        """判断是否应该使用专家模块"""
        experts = cls.choose_experts_for_dataset(dataset_name)
        return len(experts) > 0
    
    @classmethod
    def get_all_expert_configs(cls, expert_names: List[str]) -> Dict[str, Dict]:
        """获取多个专家模块的配置"""
        configs = {}
        for expert_name in expert_names:
            configs[expert_name] = cls.get_expert_config(expert_name)
        return configs


def main():
    """测试函数"""
    test_datasets = ["docvqa", "chartqa", "scienceqa", "gqa"]
    
    for dataset in test_datasets:
        experts = ExpertChooser.choose_experts_for_dataset(dataset)
        use_experts = ExpertChooser.should_use_experts(dataset)
        configs = ExpertChooser.get_all_expert_configs(experts)
        
        print(f"数据集: {dataset}")
        print(f"  推荐专家: {experts}")
        print(f"  使用专家: {use_experts}")
        print(f"  专家配置: {configs}")
        print()


if __name__ == "__main__":
    main()