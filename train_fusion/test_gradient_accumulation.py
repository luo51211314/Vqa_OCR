import torch
import numpy as np

# 模拟配置
config = {
    'training_config': {
        'gradient_accumulation_steps': 4
    }
}

# 模拟模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.mean(x)

# 初始化模型、优化器和数据
model = SimpleModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def generate_batch():
    return torch.randn(2, 10).cuda()

# 简化版训练循环
print("开始测试梯度累积更新流程...")
print(f"梯度累积步数: {config['training_config']['gradient_accumulation_steps']}")

# 模拟4个批次的训练
for step in range(8):  # 8个步骤，应该有2次梯度累积更新
    # 模拟数据
    batch = generate_batch()
    
    # 前向传播
    loss = model(batch)
    loss = loss / config['training_config']['gradient_accumulation_steps']
    
    # 反向传播
    loss.backward()
    
    # 梯度累积
    if (step + 1) % config['training_config']['gradient_accumulation_steps'] == 0 or step == 7:
        print(f"步骤 {step}: 执行梯度累积更新")
        
        # 梯度裁剪和优化器更新
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"step {step}执行完毕")

print("测试完成！")