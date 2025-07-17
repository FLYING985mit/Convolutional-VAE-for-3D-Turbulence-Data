import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os

# 新增：导入 matplotlib
import matplotlib.pyplot as plt

# ===============================================================
#  Config 和 CFD3DDataset 类 (与之前相同)
# ===============================================================

class config:
    # 路径和设备设置
    dir = "./100"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型和训练参数
    epochs = 100
    batch_size = 16
    learning_rate = 1e-3
    
    # 可视化参数
    slice_to_view = 11 # 查看第12层切片 (索引从0开始)

    # Dataset 相关参数
    image_size = 32
    list_num = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    list_Folder = [43, 55, 64, 73, 85, 97, 103, 112, 121, 127, 136, 145, 154, 163, 169, 178, 188, 202, 213, 225]
    list_time_steps = [1, 2, 3, 4, 5]


class CFD3DDataset(Dataset):
    # ... 此处代码与您之前提供的一致，无需改动 ...
    def __init__(self, img_dir, list_num=None, list_Folder=None, list_time_steps=None, global_min_val=None, global_max_val=None):
        self.dir = img_dir
        self.folder_ids = list_Folder if list_Folder is not None else config.list_Folder
        self.num_ids = list_num if list_num is not None else config.list_num
        self.time_step_ids = list_time_steps if list_time_steps is not None else config.list_time_steps
        
        self.sample_identifiers = []
        for folder_id in self.folder_ids:
            for time_step_id in self.time_step_ids:
                self.sample_identifiers.append((folder_id, time_step_id))
        
        self.data_len = len(self.sample_identifiers)

        self.global_min_val = global_min_val
        self.global_max_val = global_max_val

        if self.global_min_val is None or self.global_max_val is None:
            print("[INFO] Computing global min/max for normalization. This might take some time...")
            all_values = []
            for folder_id, time_step_id in self.sample_identifiers:
                for num_id_z in self.num_ids:
                    file_path = os.path.join(self.dir, str(folder_id), 
                                             f"Urban.pg.t{time_step_id:03d}_{folder_id}.r{num_id_z:03d}.p2m")
                    if not os.path.exists(file_path):
                        print(f"Warning: File not found, skipping: {file_path}")
                        continue 
                    
                    df = pd.read_csv(file_path, sep=' ', skiprows=2)
                    matrix = df.pivot(index='<Y(m)>', columns='<X(m)>', values='<PathGain(dB)>').values
                    all_values.extend(matrix.flatten().tolist())

            if not all_values:
                raise RuntimeError("No data values collected. Check file paths and data loading.")

            self.global_min_val = np.min(all_values)
            self.global_max_val = np.max(all_values)
            print(f"[INFO] Global min/max computed: [{self.global_min_val}, {self.global_max_val}]")
        else:
            print(f"[INFO] Using provided global normalization range: [{self.global_min_val}, {self.global_max_val}]")

        print(f"[INFO] CFD3DDataset initialized with {self.data_len} samples.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        folder_id, time_step_id = self.sample_identifiers[idx]
        matrices = []

        for num_id_z in self.num_ids:
            file_path = os.path.join(self.dir, str(folder_id), 
                                     f"Urban.pg.t{time_step_id:03d}_{folder_id}.r{num_id_z:03d}.p2m")
            df = pd.read_csv(file_path, sep=' ', skiprows=2)
            matrix = df.pivot(index='<Y(m)>', columns='<X(m)>', values='<PathGain(dB)>').values
            matrices.append(matrix)
        
        data_np = np.array(matrices, dtype=np.float32) 

        if (self.global_max_val - self.global_min_val) == 0:
            data_normalized = np.zeros_like(data_np)
        else:
            data_normalized = (data_np - self.global_min_val) / (self.global_max_val - self.global_min_val)
        
        data_scaled = data_normalized * 2.0 - 1.0
        
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0) 
        
        return data_tensor

# ===============================================================
#  3D 卷积自编码器模型 (与之前相同)
# ===============================================================
class Autoencoder3D(nn.Module):
    # ... 此处代码与之前相同，无需改动 ...
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        latent_features = self.encoder(x)
        reconstructed_x = self.decoder(latent_features)
        return reconstructed_x

# ===============================================================
#  训练函数 (稍作修改，返回训练好的模型和数据集)
# ===============================================================
def train_ae():
    print(f"Using device: {config.device}")

    # 1. 准备数据集
    full_dataset = CFD3DDataset(img_dir=config.dir)
    train_loader = DataLoader(dataset=full_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2. 初始化模型、损失函数和优化器
    model = Autoencoder3D().to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print("Starting training...")
    # 3. 训练循环
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for data_batch in train_loader:
            data_batch = data_batch.to(config.device)
            reconstructed_batch = model(data_batch)
            loss = criterion(reconstructed_batch, data_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.epochs}], Average Loss: {avg_loss:.6f}")
        
    print("Training finished.")
    # 返回训练好的模型和完整的数据集对象（用于获取全局min/max）
    return model, full_dataset


# ===============================================================
#  新增：可视化函数
# ===============================================================
def visualize_reconstruction(model, dataset, device, num_images=4):
    """
    可视化原始图像和重建图像的切片对比。

    Args:
        model (nn.Module): 训练好的自编码器模型。
        dataset (Dataset): 数据集对象，用于获取样本和归一化参数。
        device (torch.device): 'cuda' or 'cpu'.
        num_images (int): 要显示对比图的数量。
    """
    print("\nGenerating visualizations...")
    model.eval()  # 设置为评估模式，这很重要！

    # 创建一个 DataLoader 用于方便地获取一批数据进行可视化
    # shuffle=False 确保每次运行时我们看到的是相同的样本
    vis_loader = DataLoader(dataset, batch_size=num_images, shuffle=False)
    
    # 从 DataLoader 中取出一批数据
    original_batch = next(iter(vis_loader))
    original_batch = original_batch.to(device)

    # 使用 with torch.no_grad() 进行推理，可以节省内存并加速
    with torch.no_grad():
        reconstructed_batch = model(original_batch)

    # 将 Tensor 移动到 CPU 并转换为 NumPy 数组
    original_np = original_batch.cpu().numpy()
    reconstructed_np = reconstructed_batch.cpu().numpy()

    # 获取全局 min/max 用于反归一化，以便显示真实的物理值
    g_min = dataset.global_min_val
    g_max = dataset.global_max_val
    
    # -- 开始绘图 --
    slice_idx = config.slice_to_view
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 4))
    fig.suptitle(f'Original vs. Reconstructed (Slice at z={slice_idx+1})', fontsize=16)

    for i in range(num_images):
        # 提取第 i 个样本的第 slice_idx 个切片
        # 形状从 (N, 1, D, H, W) -> (H, W)
        original_slice = original_np[i, 0, slice_idx, :, :]
        recon_slice = reconstructed_np[i, 0, slice_idx, :, :]
        
        # --- 反归一化 ---
        # 1. 从 [-1, 1] 转换回 [0, 1]
        original_slice = (original_slice + 1) / 2.0
        recon_slice = (recon_slice + 1) / 2.0
        # 2. 从 [0, 1] 转换回原始数据范围
        original_slice = original_slice * (g_max - g_min) + g_min
        recon_slice = recon_slice * (g_max - g_min) + g_min

        # 绘图：原始图像切片
        ax1 = axes[i, 0]
        im1 = ax1.imshow(original_slice, cmap='viridis')
        ax1.set_title(f'Original Sample #{i+1}')
        ax1.axis('off') # 关闭坐标轴
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 绘图：重建图像切片
        ax2 = axes[i, 1]
        im2 = ax2.imshow(recon_slice, cmap='viridis', vmin=np.min(original_slice), vmax=np.max(original_slice))
        ax2.set_title(f'Reconstructed Sample #{i+1}')
        ax2.axis('off') # 关闭坐标轴
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
    # 保存图像文件
    plt.savefig("./png/reconstruction_comparison.png")
    print("Visualization saved as 'reconstruction_comparison.png'")
    plt.show()


# ===============================================================
#  主执行流程
# ===============================================================
if __name__ == '__main__':
    # 确保您的数据目录存在
    if not os.path.exists(config.dir):
        print(f"Error: Data directory not found at '{config.dir}'")
    else:
        # 1. 训练模型
        trained_model, full_dataset = train_ae()
        
        # 2. 在训练结束后进行可视化
        visualize_reconstruction(trained_model, full_dataset, config.device, num_images=4)