import torch
import numpy as np

class AdamWManualSimulator:
    def __init__(self, model, optimizer, target_param, target_param_name):
        self.model = model
        self.optimizer = optimizer
        self.state_dict = optimizer.state_dict()
        self.print_interval = 1  # 打印间隔
        self.target_param = target_param
        self.target_param_name = target_param_name
    
    def get_optimizer_params(self):
        """从optimizer中提取AdamW参数"""
        if self.target_param is None:
            print(f"未找到目标参数 {self.target_param_name}")
            return None
        
        # 使用id()进行引用比较，避免张量元素级比较导致的维度不匹配错误
        target_param_id = id(self.target_param)
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if id(param) == target_param_id:
                    return {
                        'lr': group['lr'],
                        'beta1': group.get('betas', (0.9, 0.999))[0],
                        'beta2': group.get('betas', (0.9, 0.999))[1],
                        'eps': group.get('eps', 1e-8),
                        'weight_decay': group.get('weight_decay', 0.0),
                        'amsgrad': group.get('amsgrad', False)
                    }
        return None
    
    def get_param_and_state(self):
        """获取目标参数及其状态"""
        if self.target_param is None:
            return None, None
        
        # 直接使用target_param从optimizer.state获取状态
        # optimizer.state字典使用参数对象作为键，所以这里不会触发张量比较
        param_state = self.optimizer.state.get(self.target_param, {})
        return self.target_param, param_state
    
    def print_tensor_stats(self, tensor, name):
        """打印张量的统计信息"""
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            print(f"⚠️ {name} 包含 {nan_count} 个 NaN 值")
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            print(f"⚠️ {name} 包含 {inf_count} 个 Inf 值")
        
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        
        print(f"{name:<20} | max: {max_val:12.8f} | min: {min_val:12.8f} | mean: {mean_val:12.8f} | std: {std_val:12.8f}")
    
    def simulate_adamw_update(self):
        """手动模拟AdamW更新过程"""
        # 获取优化器参数
        opt_params = self.get_optimizer_params()
        if not opt_params:
            print("未找到ocr_text_projector.weight参数")
            return
        
        # 打印优化器参数
        print("=== AdamW 优化器参数 ===")
        for key, value in opt_params.items():
            print(f"{key}: {value}")
        print()
        
        # 获取参数和状态
        param, param_state = self.get_param_and_state()
        if param is None:
            print("未找到ocr_text_projector.weight参数")
            return
        
        # 初始化状态（如果不存在）
        if 'step' not in param_state:
            param_state['step'] = 0
        if 'exp_avg' not in param_state:
            param_state['exp_avg'] = torch.zeros_like(param)
        if 'exp_avg_sq' not in param_state:
            param_state['exp_avg_sq'] = torch.zeros_like(param)
        if opt_params['amsgrad'] and 'max_exp_avg_sq' not in param_state:
            param_state['max_exp_avg_sq'] = torch.zeros_like(param)
        
        # 记录初始权重
        initial_weight = param.data.clone()
        
        # 打印初始权重统计
        print("=== 更新前权重统计 ===")
        self.print_tensor_stats(param.data, "初始权重")
        
        # 如果有梯度，打印梯度统计
        if param.grad is not None:
            print("=== 梯度统计 ===")
            self.print_tensor_stats(param.grad.data, "梯度")
        else:
            print("⚠️  未找到梯度信息")
            return
        
        # 模拟AdamW更新步骤
        print("\n=== 模拟AdamW更新过程 ===")
        
        # 步骤1: 更新step计数器
        param_state['step'] += 1
        step = param_state['step']
        print(f"步骤1: 更新step计数器 -> {step}")
        
        # 步骤2: 应用权重衰减
        if opt_params['weight_decay'] > 0:
            print(f"步骤2: 应用权重衰减 (weight_decay={opt_params['weight_decay']})")
            grad = param.grad.data.add(param.data, alpha=opt_params['weight_decay'])
        else:
            grad = param.grad.data.clone()
        
        # 步骤3: 更新一阶矩估计 exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg = param_state['exp_avg']
        exp_avg.mul_(opt_params['beta1']).add_(grad, alpha=1 - opt_params['beta1'])
        print("步骤3: 更新一阶矩估计")
        self.print_tensor_stats(exp_avg, "exp_avg")
        
        # 步骤4: 更新二阶矩估计 exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        exp_avg_sq = param_state['exp_avg_sq']
        exp_avg_sq.mul_(opt_params['beta2']).addcmul_(grad, grad, value=1 - opt_params['beta2'])
        print("步骤4: 更新二阶矩估计")
        self.print_tensor_stats(exp_avg_sq, "exp_avg_sq")
        
        # 步骤5: 计算偏差校正项
        bias_correction1 = 1 - opt_params['beta1'] ** step
        bias_correction2 = 1 - opt_params['beta2'] ** step
        print(f"步骤5: 计算偏差校正项 -> bias_correction1={bias_correction1:.8f}, bias_correction2={bias_correction2:.8f}")
        
        # 步骤6: 计算学习率缩放因子
        lr = opt_params['lr'] * np.sqrt(bias_correction2) / bias_correction1
        print(f"步骤6: 计算学习率缩放因子 -> lr={lr:.8f}")
        
        # 步骤7: 计算分母 denom = sqrt(exp_avg_sq) + eps
        denom = exp_avg_sq.sqrt().add_(opt_params['eps'])
        print("步骤7: 计算分母 denom")
        self.print_tensor_stats(denom, "denom")
        
        # 步骤8: 应用更新 w = w - lr * exp_avg / denom
        print("步骤8: 应用参数更新")
        # 创建更新量的副本用于分析
        update = exp_avg.div(denom).mul_(lr)
        self.print_tensor_stats(update, "更新量")
        
        # 手动更新参数（但不实际修改原参数）
        updated_weight = param.data.clone().sub_(update)
        
        # 打印更新后的权重统计
        print("\n=== 模拟更新后权重统计 ===")
        self.print_tensor_stats(updated_weight, "更新后权重")
        
        # 计算权重变化
        weight_diff = updated_weight - initial_weight
        print("=== 权重变化统计 ===")
        self.print_tensor_stats(weight_diff, "权重变化")
        
        # 检查更新后是否有NaN或Inf
        has_nan = torch.isnan(updated_weight).any().item()
        has_inf = torch.isinf(updated_weight).any().item()
        
        if has_nan or has_inf:
            print("\n❌ 警告: 模拟更新后的权重包含异常值!")
            # 找出异常值的位置
            if has_nan:
                nan_positions = torch.nonzero(torch.isnan(updated_weight), as_tuple=False)
                print(f"NaN值位置 (前10个): {nan_positions[:10]}")
            if has_inf:
                inf_positions = torch.nonzero(torch.isinf(updated_weight), as_tuple=False)
                print(f"Inf值位置 (前10个): {inf_positions[:10]}")
            
            # 打印这些位置的中间计算值
            if has_nan or has_inf:
                # 选择一个异常位置进行详细分析
                if has_nan:
                    sample_pos = nan_positions[0].tolist()
                else:
                    sample_pos = inf_positions[0].tolist()
                
                print(f"\n异常位置 {sample_pos} 的详细计算值:")
                print(f"原始权重: {initial_weight[tuple(sample_pos)]}")
                print(f"梯度: {param.grad[tuple(sample_pos)]}")
                print(f"exp_avg: {exp_avg[tuple(sample_pos)]}")
                print(f"exp_avg_sq: {exp_avg_sq[tuple(sample_pos)]}")
                print(f"denom: {denom[tuple(sample_pos)]}")
                print(f"更新量: {update[tuple(sample_pos)]}")
                print(f"更新后权重: {updated_weight[tuple(sample_pos)]}")
        else:
            print("\n✅ 模拟更新后的权重正常，无NaN或Inf值")
    
    def run(self):
        """运行模拟器"""
        print(f"开始模拟AdamW对{self.target_param_name}的更新过程...")
        print("="*80)
        self.simulate_adamw_update()
        print("="*80)
        print("模拟完成")

# 使用示例（在train_script.py中调用）
"""
# 在train_script.py中的optimizer.step()前添加以下代码
from debug_adamw_projector import AdamWManualSimulator

# 为模拟器准备目标参数
target_param = None
target_param_name = None
for name, p in self.model.named_parameters():
    if 'ocr_text_projector.weight' in name:
        target_param = p
        target_param_name = name
        break

# 创建并运行模拟器
simulator = AdamWManualSimulator(self.model, self.optimizer, target_param, target_param_name)
simulator.run()

# 然后再执行正常的optimizer.step()
self.optimizer.step()
"""