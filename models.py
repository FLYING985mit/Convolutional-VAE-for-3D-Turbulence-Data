import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# class UnFlatten(nn.Module):
#     # NOTE: (size, x, x, x) are being computed manually as of now (this is based on output of encoder)
#     def forward(self, input, size=512): # size=128
#         return input.view(input.size(0), size, 3, 3, 3)
#         # return input.view(input.size(0), size, 6, 6, 6)

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super(UnFlatten, self).__init__()
        self.target_shape = target_shape # target_shape is a tuple like (channels, D, H, W)

    def forward(self, input):
        # input.size(0) is batch_size
        return input.view(input.size(0), *self.target_shape) # Unpack target_shape tuple

class CVAE_3D(nn.Module):
    def __init__(self, image_channels=1, z_dim=1024): # image_channels 改为1，z_dim可以设置为1024
        super(CVAE_3D, self).__init__()
        print()
        print("[INFO] instantiating pytorch model: 3D CVAE (Increased Width)")

        # Encoder (调整 stride 和 padding, 增加 out_channels)
        self.encoder = nn.Sequential(
            # Input: (C, 32, 32, 32)
            nn.Conv3d(in_channels=image_channels, out_channels=64, kernel_size=4, stride=1, padding=0), # Output: (64, 29, 29, 29)
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4,stride=1, padding=0), # Output: (128, 26, 26, 26)
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0), # Output: (256, 26, 26, 26)
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0), # Output: (512, 26, 26, 26)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0), # Output: (512, 23, 23, 23)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(), # 原始代码这里有 ReLU

            # nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=4, stride=1, padding=0), # Output: (1024, 20, 20, 20)
            # nn.BatchNorm3d(num_features=1024),
            # nn.ReLU(), # 原始代码这里有 ReLU

            # nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=4, stride=1, padding=0), # Output: (1024, 17, 17, 17)
            # nn.BatchNorm3d(num_features=1024),
            # nn.ReLU(), # 原始代码这里有 ReLU

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=6, stride=1, padding=0), # Output: (512, 14, 14, 14)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(), # 原始代码这里有 ReLU

             nn.Conv3d(in_channels=512, out_channels=512, kernel_size=8, stride=1, padding=0), # Output: (512, 10, 10, 10)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(), # 原始代码这里有 ReLU

             nn.Conv3d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0), # Output: (1024, 7, 7, 7)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(), # 原始代码这里有 ReLU
            
            Flatten() # reshape layer (output will be 1024)
        )

        # fully connected layers to compute mu and sigma
        self.encoder_final_flat_dim = 4096 # Flatten 后的维度现在是 1024
        self.fc1 = nn.Linear(self.encoder_final_flat_dim, z_dim)
        self.fc2 = nn.Linear(self.encoder_final_flat_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, self.encoder_final_flat_dim) # 从 z_dim 映射回 Flatten 后的维度

        # Decoder (调整 stride 和 padding, 确保 output_padding 使得尺寸正确翻倍)
        self.decoder = nn.Sequential(
            UnFlatten(target_shape=(512, 2, 2, 2)), # 对应编码器 Flatten 前的维度
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (1024, 1, 1, 1)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),

            nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=8, stride=1, padding=0, output_padding=0), # Output: (1024, 1, 1, 1)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),

            # nn.ConvTranspose3d(in_channels=1024, out_channels=1024, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (512, 2, 2, 2)
            # nn.BatchNorm3d(num_features=1024),
            # nn.ReLU(),

            # nn.ConvTranspose3d(in_channels=1024, out_channels=1024, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (512, 2, 2, 2)
            # nn.BatchNorm3d(num_features=1024),
            # nn.ReLU(),

            nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=6, stride=1, padding=0, output_padding=0), # Output: (512, 2, 2, 2)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),

            nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (512, 2, 2, 2)
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (256, 4, 4, 4)
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (128, 8, 8, 8)
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (64, 16, 16, 16)
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            
            # 最后一层：从 64 通道到 image_channels (1)
            nn.ConvTranspose3d(in_channels=64, out_channels=image_channels, kernel_size=4, stride=1, padding=0, output_padding=0), # Output: (image_channels, 32, 32, 32)
            
            # === 激活函数根据你的数据归一化范围选择 ===
            # 如果数据归一化到 [-1, 1]，加 Tanh：
            nn.Tanh(), 
            # 如果数据归一化到 [0, 1]，加 Sigmoid：
            # nn.Sigmoid(),
            # 移除 BatchNorm3d，因为这是最后一层输出，通常不需要
        )

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-5.0, max=5.0) # 或者更严格如 min=-20.0, max=2.0
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        # print(11111)
        h = self.encoder(x)
        # print("[INFO] h size:", h.size()) # torch.Size([10, 1024])
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        z_representation = self.representation(x)
        z, mu, logvar = self.encode(x)
        z = self.fc3(z) # z 经过 fc3 变换后，形状与 encoder_final_flat_dim 相同，才能送入 UnFlatten

        return self.decode(z), mu, logvar, z_representation



class CVAE_3D_II(nn.Module):
    def __init__(self, image_channels=3, h_dim=128, z_dim=32):
        super(CVAE_3D_II, self).__init__()
        print()
        print("[INFO] instantiating pytorch model: 3D CVAE")

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=32, kernel_size=4, stride=1, padding=0), 
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            Flatten() # reshape layer
        )

        # fully connected layers to compute mu and sigma
        # z_dim is set by user
        # h_dim should be computed manually based on output of convs
        self.fc1 = nn.Linear(13824, z_dim)
        self.fc2 = nn.Linear(13824, z_dim)
        # self.fc1 = nn.Linear(h_dim, z_dim)
        # self.fc2 = nn.Linear(h_dim, z_dim)

        # self.fc3 = nn.Linear(z_dim, h_dim) # dense layer to connect to decoder
        self.fc3 = nn.Linear(z_dim, 13824)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=1, padding=0), # dimensions should be as original
            nn.BatchNorm3d(num_features=3))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # std = logvar.mul(0.5).exp_()
        # eps = torch.randn(*mu.size())
        eps = torch.rand_like(std)
        # z = mu + std * eps
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        # print("[INFO] bottleneck h size:", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        print("[INFO] h size:", h.size()) # torch.Size([10, 27648])
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        # print("[INFO] Input data shape:", x.size())

        # Step 1: compute representation (fetch it separately for later clustering)
        z_representation = self.representation(x)
        print("[INFO] Forward z_representation:", z_representation.size())
        # print("[INFO] Reshaped latent z", z_representation.view(z_representation.size(0), 8, 8).size())

        # Step 2: call full CVAE --> encode & decode
        z, mu, logvar = self.encode(x)
        z = self.fc3(z)
        # print("[INFO] Latent z after dense fc:", z.size())
        # print("[INFO] mu:", mu.size())
        # print("[INFO] logvar", logvar.size())

        return self.decode(z), mu, logvar, z_representation
