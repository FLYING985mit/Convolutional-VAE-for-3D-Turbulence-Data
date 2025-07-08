# import torch
# # import torch.nn as nn
# # from torchvision.utils import save_image, make_grid
# from sklearn.manifold import TSNE, SpectralEmbedding
# from sklearn.decomposition import TruncatedSVD

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# def init_weights(model):
#     """
#     Set weight initialization for Conv3D in network.
#     Based on: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24
#     """
#     if isinstance(model, torch.nn.Conv3d):
#         torch.nn.init.xavier_uniform_(model.weight)
#         torch.nn.init.constant_(model.bias, 0)
#         # torch.nn.init.zeros_(model.bias)

# def plot_cube_slice(cube_sample, z_slice, channel):
#     """
#     Plots a horizontal slice of a cube in the z direction.
#     """
#     plt.imshow(cube_sample[:,:,z_slice,channel]) # [sample_idx] (x,y,z,channel)
#     plt.colorbar()
#     plt.plot()

# def show_histogram(values, norm_func):
#     """
#     """
#     print(values.shape)
#     n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])

#     for c, p in zip(norm_func(bin_centers), patches):
#         plt.setp(p, 'facecolor', cm.viridis(c))

#     plt.show()

# def plot_cube(cube, IMG_DIM, norm_func, angle=320):
#     """
#     """
#     # right now it works better if normalize again here
#     cube = norm_func(cube)

#     # apply heatmap, the object viridis is a callable,
#     # that when passed a float between 0 and 1 returns an RGBA value from the colormap
#     facecolors = cm.viridis(cube)

#     # the filled array tells matplotlib what to fill (any true value is filled)
#     filled = facecolors[:,:,:,-1] != 0
#     x, y, z = np.indices(np.array(filled.shape) + 1)

#     # define 3D plotting
#     fig = plt.figure(figsize=(30/2.54, 30/2.54))
#     ax = fig.gca(projection='3d')
#     ax.view_init(30, angle)
#     ax.set_xlim(right=IMG_DIM+2)
#     ax.set_ylim(top=IMG_DIM+2)
#     ax.set_zlim(top=IMG_DIM+2)

#     # ax.scatter(x, y, z, filled, facecolors=facecolors, shade=False)
#     ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
#     plt.show()

# def plot_reconstruction_grid(original, reconstruction, channels, cube_dim, epoch, save_grid=False, reference_batch=False):
#     """
#     Plots the 3 channels (velocity components) of the original and reconstruction
#     cubes in a grid.

#     original: torch tensor torch.Size([128, 3, 21, 21, 21])
#     reconstruction: torch tensor torch tensor torch.Size([128, 3, 21, 21, 21])
#     """

#     # move: cuda tensor --> cpu tensor --> numpy array
#     original = original.cpu().detach().numpy()
#     reconstruction = reconstruction.cpu().detach().numpy()

#     batch_indices = [0, 1, 2, 3] # plot grid for different batch samples
#     # idx = 0 # choose one sample (cube) from 128 batch to plot

#     for index in batch_indices:
#         original_sample = original[index]
#         reconstruction_sample = reconstruction[index]

#         # swap axes from torch (3, 21, 21, 21) to numpy (21, 21, 21, 3)
#         original_sample = np.transpose(original_sample, (3, 1, 2, 0))
#         reconstruction_sample = np.transpose(reconstruction_sample, (3, 1, 2, 0))

#         # locate reconstruction-original cubes in dictionary
#         cube_samples = {"Orig" : original_sample,
#                         "Recon" : reconstruction_sample}

#         # dictionary for velocity channels subtitles
#         velocity_channels = {0 : "U0", 1 : "U1", 2 : "U2"}

#         # set figure dimensions
#         fig = plt.figure(figsize=(40, 40))

#         subplot_count = 1

#         for key, cube_sample in cube_samples.items():
#             for channel in range(channels):
#                 # define subplots
#                 ax = fig.add_subplot(2, 3, subplot_count, projection="3d")

#                 # generate heat map for each channel
#                 facecolors = cm.viridis(cube_sample[:,:,:,channel])

#                 # the filled array tells matplotlib what to fill (any true value is filled)
#                 filled = facecolors[:,:,:,-1] != 0
#                 x, y, z = np.indices(np.array(filled.shape) + 1)

#                 # define 3D plotting
#                 ax = fig.gca(projection="3d")
#                 ax.view_init(elev=30, azim=320)
#                 ax.set_xlim(right=cube_dim + 2)
#                 ax.set_ylim(top=cube_dim + 2)
#                 ax.set_zlim(top=cube_dim + 2)
#                 ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
#                 subplot_count += 1

#                 # set titles, axis fontsizes
#                 ax.set_title(key + " " + "vel component " + velocity_channels[channel], fontsize=55)
#                 ax.tick_params(axis="both", which='major', labelsize=30)

#         if save_grid:
#             if reference_batch:
#                 plt.savefig("results/REFERENCE_original_reconstruction_grid_epoch_{}_batchid_{}.png".format(epoch, index))
#                 plt.close()
#             else:
#                 plt.savefig("results/RANDOM_original_reconstruction_grid_epoch_{}_batchid_{}.png".format(epoch, index))
#                 plt.close()
#         else:
#             plt.show()

# def plot_generation_grid(model, device, grid_size=9, save_grid=False):
#     """
#     """
#     with torch.no_grad():
#         samples = torch.randn(grid_size, 27648).to(device)
#         # samples = torch.randn(32, 128).to(device)
#         samples = model.decode(samples).cpu() # returns a torch.Size([9, 3, 21, 21, 21])
#         print("samples decoded", samples.size())

#         # grid = make_grid(sample)
#         # writer.add_image('sampling', grid, epoch)
#         # save_image(sample.view(64, 4, 41, 41), "results/samples_" + str(epoch) + ".png")

# # def save_representations(latent_batch, epoch, writer, args):
# #     """
# #     """
# #     # print("latent batch size:", latent_batch.size())
# #     nrow = min(latent_batch.size(0), 8)
# #     grid = make_grid(latent_batch.view(args.batch_size, 1, 8, 8)[:nrow].cpu(), nrow=nrow, normalize=True)
# #     writer.add_image("latent representations", grid, epoch)
# #     save_image(grid.cpu(), "results/representations_" + str(epoch) + ".png", nrow=nrow)
# #
# # def save_projected_representations(latent_batch, epoch, writer, args, download=True):
# #     """
# #     Projects latent vectors in 2D using PCA and t-SNE.
# #     """
# #     print("latent batch size:", latent_batch.size()) # (128, 64)
# #     np_latent_batch = latent_batch.cpu().numpy()
# #     print("np latent batch size:", np_latent_batch.shape)
# #
# #     PCA_latent_batch = TruncatedSVD(n_components=3).fit_transform(np_latent_batch)
# #     tSNE_latent_batch = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(np_latent_batch)
# #     print("PCA latent batch shape:", PCA_latent_batch.shape)
# #
# #     # plot PCA
# #     plt.rcParams.update({'font.size': 22})
# #     fig = plt.figure(figsize=(12,12))
# #     ax1 = fig.add_subplot(111)
# #     ax1.scatter(PCA_latent_batch[:,0], PCA_latent_batch[:,1])
# #     plt.title('PCA on latent represenations', fontdict = {'fontsize' : 30})
# #     plt.xlabel("Principal Component 1", fontsize=22)
# #     plt.ylabel("Principal Component 2", fontsize=22)
# #     plt.legend()
# #     plt.grid()
# #
# #     if download:
# #         plt.savefig("results/projected_representations_" + str(epoch) + "_pca.png")
# #     else:
# #         plt.show()
# #
# #     # plot t-SNE
# #     plt.rcParams.update({'font.size': 22})
# #     fig = plt.figure(figsize=(12,12))
# #     ax1 = fig.add_subplot(111)
# #     ax1.scatter(tSNE_latent_batch[:,0], tSNE_latent_batch[:,1])
# #     plt.title('t-SNE on latent represenations', fontdict = {'fontsize' : 30})
# #     plt.legend()
# #     plt.grid()
# #
# #     if download:
# #         plt.savefig("results/projected_representations_" + str(epoch) + "_tsne.png")
# #     else:
# #         plt.show()









import torch
# from torchvision.utils import save_image, make_grid # 如果不需要 make_grid 和 save_image，可以不导入
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # 只有在需要绘制 3D 体素图时才需要
from matplotlib import cm # 用于颜色映射

def init_weights(model):
    """
    Set weight initialization for Conv3D in network.
    Based on: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24
    """
    if isinstance(model, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.constant_(model.bias, 0)
        # torch.nn.init.zeros_(model.bias)

# 原始的 plot_cube_slice 和 show_histogram 保持不变或根据需要修改
# 这些函数在 plot_reconstruction_grid 内部未使用
# 如果你的数据是 (C, D, H, W) 且 C=1，那么 plot_cube_slice 需要这样：
def plot_cube_slice(cube_sample_tensor, z_slice, channel_idx=0):
    """
    Plots a horizontal slice of a cube in the z direction.
    cube_sample_tensor: torch tensor of shape (C, D, H, W) for single sample
    """
    # 确保 cube_sample_tensor 形状是 (C, D, H, W)
    if cube_sample_tensor.ndim == 4: # 如果是 (C, D, H, W)
        cube_sample_np = cube_sample_tensor[channel_idx, :, :, :].cpu().numpy() # 取出单个通道并转为 NumPy
    elif cube_sample_tensor.ndim == 3: # 如果是 (D, H, W)
        cube_sample_np = cube_sample_tensor.cpu().numpy()
    else:
        raise ValueError("Input cube_sample_tensor must be 3D (D,H,W) or 4D (C,D,H,W)")
    
    plt.imshow(cube_sample_np[z_slice, :, :], cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title(f"Slice at Z={z_slice} (Channel {channel_idx})")
    plt.show()

# plot_cube 函数 (如果用于单独绘制一个 3D 体素，需要引入 Axes3D)
# 如果使用此函数，请确保在文件顶部取消注释 `from mpl_toolkits.mplot3d import Axes3D`
def plot_cube(cube, IMG_DIM, cmap_obj=cm.viridis, angle=320): # 修改 norm_func 为 cmap_obj，因为它是颜色映射对象
    """
    Plots a 3D voxel cube.
    cube: NumPy array of shape (D, H, W) for single channel
    """
    # 确保 cube 是单通道的 (D, H, W)
    if cube.ndim == 4 and cube.shape[0] == 1: # 如果是 (1, D, H, W)
        cube = cube[0, :, :, :]
    elif cube.ndim != 3:
        raise ValueError("Input cube must be 3D (D,H,W) or 4D (1,D,H,W)")

    # 归一化到 [0, 1] 范围以便颜色映射
    normalized_cube = (cube - np.min(cube)) / (np.max(cube) - np.min(cube)) if np.max(cube) - np.min(cube) != 0 else np.zeros_like(cube)

    # 应用颜色映射
    facecolors = cmap_obj(normalized_cube) 

    filled = facecolors[:,:,:,-1] != 0 # Alpha channel check, assuming non-zero alpha means filled
    x, y, z = np.indices(np.array(filled.shape) + 1)

    fig = plt.figure(figsize=(10, 10)) 
    ax = fig.gca(projection='3d') # 创建 3D 投影
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM) 
    ax.set_ylim(top=IMG_DIM)
    ax.set_zlim(top=IMG_DIM)
    ax.set_aspect('auto') 

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()


# === 核心修改：plot_reconstruction_grid ===
def plot_reconstruction_grid(original, reconstruction, channels, cube_dim, epoch, save_grid=False, is_reference_batch=False):
    """
    Plots a grid of original and reconstructed 3D cubes (Z-axis slices).
    Adapted for single-channel data.

    original: torch tensor torch.Size([Batch, 1, D, H, W])
    reconstruction: torch tensor torch.Size([Batch, 1, D, H, W])
    """

    # move: cuda tensor --> cpu tensor --> numpy array
    original_np = original.cpu().detach().numpy() # Shape (Batch, 1, D, H, W)
    reconstruction_np = reconstruction.cpu().detach().numpy() # Shape (Batch, 1, D, H, W)

    batch_indices = [0, 1, 2, 3] # 绘制前 4 个样本
    if len(original_np) < max(batch_indices) + 1: # 如果batch_size太小，调整索引
        batch_indices = list(range(len(original_np)))

    # 选择一个有代表性的 Z 轴切片，例如中间切片
    z_slice_to_plot = cube_dim // 2 

    # 创建子图网格：行数是样本数，列数是 2 (原始 vs 重建)
    fig, axes = plt.subplots(len(batch_indices), 2, figsize=(8, 4 * len(batch_indices))) 
    fig.suptitle(f"Epoch {epoch} Reconstructions (Z-slice: {z_slice_to_plot})", fontsize=16)

    # 确保 axes 是一个 2D 数组，即使只有一行 (len(batch_indices)=1)
    if len(batch_indices) == 1:
        axes = np.expand_dims(axes, axis=0) # 将 (ax_orig, ax_recon) 变为 [[ax_orig, ax_recon]]

    for i, index in enumerate(batch_indices):
        original_sample_slice = original_np[index, 0, z_slice_to_plot, :, :] # 取出第 0 个通道的 Z 切片
        reconstruction_sample_slice = reconstruction_np[index, 0, z_slice_to_plot, :, :] # 取出第 0 个通道的 Z 切片

        # 绘制原始切片
        ax_orig = axes[i, 0]
        im_orig = ax_orig.imshow(original_sample_slice, cmap='viridis', origin='lower')
        ax_orig.set_title(f"Original Sample {index}") # 移除 Z-slice from title, already in suptitle
        ax_orig.axis('off')
        plt.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)

        # 绘制重建切片
        ax_recon = axes[i, 1]
        im_recon = ax_recon.imshow(reconstruction_sample_slice, cmap='viridis', origin='lower')
        ax_recon.set_title(f"Reconstruction Sample {index}") # 移除 Z-slice from title
        ax_recon.axis('off')
        plt.colorbar(im_recon, ax=ax_recon, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，为suptitle留出空间

    if save_grid:
        filename = "results/"
        if is_reference_batch: # 参数名从 reference_batch 改为 is_reference_batch 更清晰
            filename += "REFERENCE_"
        else:
            filename += "RANDOM_"
        filename += "reconstruction_grid_epoch_{}.png".format(epoch) # 移除 batchid
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

# === 核心修改：plot_generation_grid ===
def plot_generation_grid(model, device, grid_size=9, save_grid=False, epoch=0, cube_dim=32, channels=1): # 添加 cube_dim, channels, epoch 参数
    """
    Generates and plots a grid of synthetic 3D cubes (Z-axis slices) from the trained model.
    """
    with torch.no_grad():
        # 你的 CVAE_3D Flatten 后的维度是 512，假设 z_dim 是 64
        # 所以这里的 samples 应该是 (grid_size, z_dim)
        # 从模型中获取 z_dim，假设 model 对象有 z_dim 属性
        latent_dim = model.z_dim # 假设你的 VAE 模型有 .z_dim 属性

        samples = torch.randn(grid_size, latent_dim).to(device) 
        
        # 确保 model.decode(samples) 返回的形状是 (Batch, C, D, H, W)
        # model.decode(z) 在 models.py 中接收 z，并返回解码结果
        generated_samples_tensor = model.decode(samples) # returns (grid_size, C, D, H, W)
        generated_samples_np = generated_samples_tensor.cpu().numpy()

        print("Generated samples shape:", generated_samples_np.shape)

        # 绘制 Z 轴切片
        z_slice_to_plot = cube_dim // 2

        fig, axes = plt.subplots(grid_size // 3, 3, figsize=(12, 4 * (grid_size // 3))) 
        fig.suptitle(f"Epoch {epoch} Generated Cubes (Z-slice: {z_slice_to_plot})", fontsize=16)

        # 确保 axes 是一个 2D 数组，即使只有一行 (grid_size // 3 == 1)
        if grid_size // 3 == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(grid_size):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # 取出单个生成样本的第 0 个通道的 Z 切片
            generated_slice = generated_samples_np[i, 0, z_slice_to_plot, :, :] 
            im = ax.imshow(generated_slice, cmap='viridis', origin='lower')
            ax.set_title(f"Generated Sample {i}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_grid:
            plt.savefig(f"results/GENERATED_grid_epoch_{epoch}.png", dpi=300)
            plt.close()
        else:
            plt.show()

# 其余的 save_representations 和 save_projected_representations 如果你需要，
# 它们也需要根据你的 z_dim 调整 make_grid 的维度和 t-SNE/PCA 的输入。
# 例如，make_grid(latent_batch.view(args.batch_size, 1, 8, 8)[:nrow].cpu(), ...
# 这里的 8,8 需要根据你的 z_dim 调整成一个能形成方形的维度，或者直接处理成 1D 散点图。