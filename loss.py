import torch
from torch.nn import functional as F
import numpy as np
import pdb


def schedule_KL_annealing(start, stop, n_epochs, n_cycle=4, ratio=0.5):
    # 这个函数可以保留，但其输出的 kl_weights 不再被 loss_function 使用
    # 在 main.py 中，你可以继续调用它，但 kl_weight 传给 loss_function 将是无关紧要的
    weights = np.ones(n_epochs)
    period = n_epochs/n_cycle
    step = (stop-start)/(period*ratio)

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epochs):
            weights[int(i+c*period)] = v
            v += step
            i += 1

    return weights

# === 修改 loss_function ===
def loss_function(recon_x, x, mu, logvar, kl_weight): # mu, logvar, kl_weight 现在是“占位符”
    """
    Computes Mean Squared Error (MSE) for a standard Autoencoder.
    """
    
    # Reconstruction loss (MSE)
    # 使用 reduction='mean'
    MSE_loss_per_batch_mean = torch.nn.MSELoss(reduction='mean')(recon_x, x)
    weighted_MSE = MSE_loss_per_batch_mean # 不需要乘数，除非你希望手动平衡，但现在不建议
    
    # KLD_loss 不再存在，设置为 0
    KLD_loss_per_batch_sum = torch.tensor(0.0, device=recon_x.device) # 或者直接返回 None
    weighted_KLD = torch.tensor(0.0, device=recon_x.device) # 同样为 0

    total_loss = weighted_MSE # 总损失现在就是重建损失

    # 返回值保持与 VAE 相同的结构 (为了兼容 train.py 和 test.py)，但 KLD 相关为 0
    return total_loss, weighted_MSE, KLD_loss_per_batch_sum



# def schedule_KL_annealing(start, stop, n_epochs, n_cycle=4, ratio=0.5):
#     """
#     Custom function for multiple annealing scheduling: Monotonic and cyclical_annealing
#     Given number of epochs, it returns the value of the KL weight at each epoch as a list.

#     Based on from: https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
#     """

#     weights = np.ones(n_epochs)
#     period = n_epochs/n_cycle
#     step = (stop-start)/(period*ratio) # linear schedule

#     for c in range(n_cycle):
#         v , i = start , 0
#         while v <= stop and (int(i+c*period) < n_epochs):
#             weights[int(i+c*period)] = v
#             v += step
#             i += 1

#     return weights

# def loss_function(recon_x, x, mu, logvar, kl_weight):
#     """
#     Computes binary cross entropy and analytical expression of KL divergence used to train Variational Autoencoders

#     Losses are calculated per batch (recon vs original). Their sizes are torch.Size([128, 3, 21, 21, 21])

#     Total loss is reconstruction + KL divergence summed over batch
#     """
    
#     # reconstruction loss (MSE/BCE for image-like data)
#     # CE = torch.nn.CrossEntropyLoss()(recon_x, x)
#     # MSE = torch.nn.MSELoss(reduction='mean')(recon_x, x)

#     MSE = 0.1 * torch.nn.MSELoss(reduction='sum')(recon_x, x)
#     # MSE =  torch.nn.MSELoss(reduction='sum')(recon_x, x)


#     # BCE = F.binary_cross_entropy(recon_x, x, reduction="mean") # only takes data in range [0, 1]
#     # BCEL = torch.nn.BCEWithLogitsLoss(reduction="mean")(recon_x, x)

#     # KL divergence loss (with annealing)
#     KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # sum or mean
#     # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     # print("KLD:", KLD)
#     KLD = KLD * kl_weight

#     return MSE + KLD, MSE, KLD
















#原版
# def schedule_KL_annealing(start, stop, n_epochs, n_cycle=4, ratio=0.5):
#     weights = np.ones(n_epochs)
#     period = n_epochs/n_cycle
#     step = (stop-start)/(period*ratio) # linear schedule

#     for c in range(n_cycle):
#         v , i = start , 0
#         while v <= stop and (int(i+c*period) < n_epochs):
#             weights[int(i+c*period)] = v
#             v += step
#             i += 1

#     return weights

# def loss_function(recon_x, x, mu, logvar, kl_weight):
#     """
#     Computes Mean Squared Error (MSE) for reconstruction and KL divergence for VAEs.
#     """
    
#     # Reconstruction loss (MSE)
#     # 0.1 * sum(MSE) / batch_size 
#     # 或者直接使用 reduction='mean' 并在外部乘以 0.1
#     # 假设你的 recon_x 和 x 的值范围在 [0,1] 或 [-1,1] 之间。
#     # 这里保持 reduction='sum' 以匹配原始代码在 main.py 中的累加和除以 dataset length 的逻辑。
#     MSE_loss_per_batch_sum = torch.nn.MSELoss(reduction='sum')(recon_x, x)
#     weighted_MSE = 0.1 * MSE_loss_per_batch_sum 

#     # KL divergence loss (analytical expression)
#     # KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # 如果希望KLD也是批次平均
#     KLD_loss_per_batch_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
#     # === 关键修正：应用 KL 退火权重 ===
#     weighted_KLD = KLD_loss_per_batch_sum * kl_weight 

#     # Total loss is weighted reconstruction + weighted KL divergence
#     total_loss = weighted_MSE + weighted_KLD

#     return total_loss, MSE_loss_per_batch_sum, KLD_loss_per_batch_sum # 返回未经权重的 MSE 和 KLD 方便查看原始数值
