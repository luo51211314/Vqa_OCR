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
    
    # 创建训练器并开始训练
    print("\n开始初始化训练器...")
    trainer = Trainer(config)
    trainer.initialize()
    
    print("\n开始训练...")
    trainer.train()
    
    print("\n训练完成！")
    
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
    
    # 执行关机命令
    shutdown_system()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n程序发生错误，不执行自动关机")
        # 确保关闭日志文件
        if 'log_fp' in locals():
            log_fp.close()