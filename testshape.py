import torch
from models import CVAE_3D # 假设你的模型定义在 models.py 中

# 实例化模型 (image_channels=1, z_dim可以随意，但建议32或64)
# 确保 UnFlatten 类的修改也应用了
model = CVAE_3D(image_channels=1, z_dim=64)
model.eval() # 设置为评估模式

# 创建一个假的输入张量
# Batch_Size=1, Channels=1, Depth=32, Height=32, Width=32
dummy_input = torch.randn(1, 1, 32, 32, 32)

# 进行前向传播
with torch.no_grad():
    reconstructed_output, mu, logvar, z_representation = model(dummy_input)

print(f"输入形状: {dummy_input.shape}")
print(f"重建输出形状: {reconstructed_output.shape}") # 期望: torch.Size([1, 1, 32, 32, 32])
print(f"潜在均值 (mu) 形状: {mu.shape}") # 期望: torch.Size([1, 64])
print(f"潜在表示 (z_representation) 形状: {z_representation.shape}") # 期望: torch.Size([1, 64])

# 检查重建输出的数值范围，如果希望在[0,1]或其他范围，可以进行验证
# print(f"重建输出值范围: [{reconstructed_output.min().item()}, {reconstructed_output.max().item()}]")