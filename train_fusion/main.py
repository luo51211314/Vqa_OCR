import os
import sys
import json
import torch
import time
import datetime
import subprocess
from train_script import Trainer

def setup_logging():
    """设置日志记录"""
    # 创建logs目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件，使用时间戳命名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")
    
    print(f"日志将保存到: {log_file}")
    
    # 创建文件对象
    log_fp = open(log_file, 'w', encoding='utf-8')
    
    # 定义日志记录类
    class Logger:
        def __init__(self, log_fp):
            self.log_fp = log_fp
        
        def write(self, message):
            sys.__stdout__.write(message)
            self.log_fp.write(message)
            self.log_fp.flush()
        
        def flush(self):
            sys.__stdout__.flush()
            self.log_fp.flush()
    
    # 重定向标准输出和错误输出
    sys.stdout = Logger(log_fp)
    sys.stderr = Logger(log_fp)
    
    return log_fp, log_file

def shutdown_system():
    """安全地关闭系统"""
    print("\n" + "="*50)
    print(f"[{datetime.datetime.now()}] 准备关闭系统...")
    try:
        # 使用shutdown命令，添加延迟以确保日志完全写入
        subprocess.run(["/usr/bin/shutdown", "-h", "now"], check=True)
        print(f"[{datetime.datetime.now()}] 关机命令已发送")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.datetime.now()}] 发送关机命令失败: {e}")

def main():
    """主入口函数"""
    # 记录开始时间
    start_time = time.time()
    print(f"[{datetime.datetime.now()}] 程序开始运行")
    
    # 设置日志记录
    log_fp, log_file = setup_logging()
    
    # 加载配置文件
    config_path = "config.json"
    # 获取训练阶段参数，默认为阶段1
    training_stage = 1
    
    # 解析命令行参数
    # 支持格式: python main.py [config_path] [stage]
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            training_stage = int(sys.argv[2])
            if training_stage not in [1, 2]:
                print(f"警告: 无效的训练阶段 {training_stage}，将使用默认值 1")
                training_stage = 1
        except ValueError:
            print(f"警告: 无法解析训练阶段参数 {sys.argv[2]}，将使用默认值 1")
    
    print(f"当前训练阶段: {training_stage}")
    
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
    
    # 检查数据集是否存在 - 支持多路径配置
    train_data_paths = config['data_config']['train_data_paths']
    val_data_paths = config['data_config']['val_data_paths']
    
    # 检查所有训练数据路径
    for path in train_data_paths:
        if not os.path.exists(path):
            print(f"错误: 训练数据集不存在: {path}")
            sys.exit(1)
    
    # 检查所有验证数据路径
    for path in val_data_paths:
        if not os.path.exists(path):
            print(f"错误: 验证数据集不存在: {path}")
            sys.exit(1)
    
    print(f"训练数据集路径: {', '.join(train_data_paths)}")
    print(f"验证数据集路径: {', '.join(val_data_paths)}")
    
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
    
    print(f"- 日志文件: {log_file}")
    print(f"- 程序完成后将自动关机")
    
    # 根据指定阶段配置训练
    config_stage = config.copy()
    config_stage['training_stage'] = training_stage
    
    # 对于阶段2，设置从阶段1的保存目录加载检查点
    if training_stage == 2:
        if 'save_dir' in config['training_config']:
            stage1_save_dir = config['training_config']['save_dir']
            stage1_checkpoint = os.path.join(stage1_save_dir, 'new_params.pth')
            config_stage['stage1_checkpoint_path'] = stage1_checkpoint
            print(f"[阶段2] 设置阶段1检查点路径: {stage1_checkpoint}")
            if os.path.exists(stage1_checkpoint):
                print(f"[阶段2] 阶段1检查点文件存在，将在初始化时自动加载")
            else:
                print(f"[阶段2] 警告: 未找到阶段1的检查点文件 {stage1_checkpoint}")
                # 尝试查找阶段1的最佳epoch目录
                stage1_dir = os.path.join(stage1_save_dir, "stage_1")
                if os.path.exists(stage1_dir):
                    print(f"[阶段2] 将尝试从阶段1的目录查找最佳模型: {stage1_dir}")
    
    print(f"\n" + "="*50)
    print(f"[阶段{training_stage}] 开始初始化训练器...")
    
    trainer = Trainer(config_stage)
    trainer.initialize()
    
    print(f"\n[阶段{training_stage}] 开始训练...")
    trainer.train()
    
    print(f"\n[阶段{training_stage}] 训练完成！")
    
    # 计算运行时间
    end_time = time.time()
    run_time = end_time - start_time
    hours, remainder = divmod(run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"[{datetime.datetime.now()}] 程序运行统计:")
    print(f"- 总运行时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print(f"- 日志已保存到: {log_file}")
    print("="*50)
    
    # 关闭日志文件
    log_fp.close()

if __name__ == "__main__":
    try:
        main()
        # 确保成功完成时返回退出码0
        sys.exit(0)
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n程序发生错误，不执行自动关机")
        # 确保关闭日志文件
        if 'log_fp' in locals():
            log_fp.close()
        # 发生异常时返回非零退出码
        sys.exit(1)