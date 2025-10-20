import os
import sys
import json
import torch
from train_script import Trainer

def main():
    """主入口函数"""
    # 加载配置文件
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"配置文件加载成功: {config_path}")
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        sys.exit(1)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU进行训练，这将非常慢！")
    else:
        device_count = torch.cuda.device_count()
        print(f"CUDA可用，设备数量: {device_count}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
    
    # 检查数据集是否存在
    train_data_path = config['data_config']['train_data_path']
    val_data_path = config['data_config']['val_data_path']
    
    if not os.path.exists(train_data_path):
        print(f"错误: 训练数据集不存在: {train_data_path}")
        sys.exit(1)
    
    if not os.path.exists(val_data_path):
        print(f"错误: 验证数据集不存在: {val_data_path}")
        sys.exit(1)
    
    print(f"训练数据集: {train_data_path}")
    print(f"验证数据集: {val_data_path}")
    
    # 检查保存目录
    save_dir = config['training_config']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"模型将保存到: {save_dir}")
    
    # 打印训练配置摘要
    print("\n训练配置摘要:")
    print(f"- 批量大小: {config['training_config']['batch_size']}")
    print(f"- 学习率: {config['training_config']['learning_rate']}")
    print(f"- 投影层学习率: {config['training_config'].get('projector_learning_rate', config['training_config']['learning_rate'] * 5)}")
    print(f"- 训练轮数: {config['training_config']['num_epochs']}")
    print(f"- 梯度累积步数: {config['training_config']['gradient_accumulation_steps']}")
    print(f"- LoRA: {'启用' if config['lora_config']['lora_enable'] else '禁用'}")
    
    if config['lora_config']['lora_enable']:
        print(f"  - LoRA r: {config['lora_config']['lora_r']}")
        print(f"  - LoRA alpha: {config['lora_config']['lora_alpha']}")
    
    # 创建训练器并开始训练
    print("\n开始初始化训练器...")
    trainer = Trainer(config)
    trainer.initialize()
    
    print("\n开始训练...")
    trainer.train()
    
    print("\n训练完成！")

if __name__ == "__main__":
    main()