import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super(UnFlatten, self).__init__()
        self.target_shape = target_shape
    def forward(self, input):
        return input.view(input.size(0), *self.target_shape)

class CVAE_3D(nn.Module): # 保持类名不变，但功能已变为 AE
    def __init__(self, image_channels=1, z_dim=1024): # z_dim 现在是潜在向量的直接维度
        super(CVAE_3D, self).__init__()
        print()
        print("[INFO] instantiating pytorch model: 3D Autoencoder (with Skip Connections)") # 更改打印信息

        # Encoder (5 layers, stride=2, padding=1)
        self.conv1_e = nn.Sequential( # Output: (64, 16, 16, 16)
            nn.Conv3d(in_channels=image_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU()
        )
        self.conv2_e = nn.Sequential( # Output: (128, 8, 8, 8)
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU()
        )
        self.conv3_e = nn.Sequential( # Output: (256, 4, 4, 4)
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU()
        )
        self.conv4_e = nn.Sequential( # Output: (512, 2, 2, 2)
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU()
        )
        # Bottleneck Encoder part (final downsample to 1x1x1)
        self.conv5_e = nn.Sequential( # Output: (1024, 1, 1, 1)
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=1024),
            nn.ReLU()
        )

        self.encoder_final_flat_dim = 1024 * 1 * 1 * 1 # Flatten 后的维度是 1024
        # self.fc1 = nn.Linear(self.encoder_final_flat_dim, z_dim) # 原始 CVAE 的 mu 层
        # self.fc2 = nn.Linear(self.encoder_final_flat_dim, z_dim) # 原始 CVAE 的 logvar 层
        self.fc_encode = nn.Linear(self.encoder_final_flat_dim, z_dim) # AE 直接输出 z

        self.fc3 = nn.Linear(z_dim, self.encoder_final_flat_dim) # 从 z_dim 映射回 Flatten 后的维度

        # Decoder (5 layers ConvTranspose3d, stride=2, padding=1)
        self.initial_d = nn.Sequential(
            UnFlatten(target_shape=(1024, 1, 1, 1)), # 从潜在空间恢复到空间维度 (1024, 1, 1, 1)
            nn.BatchNorm3d(num_features=1024),
            nn.ReLU()
        )
        
        self.conv1_d = nn.Sequential( 
            nn.ConvTranspose3d(in_channels=1024 + 1024, out_channels=512, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU()
        )
        self.conv2_d = nn.Sequential( 
            nn.ConvTranspose3d(in_channels=512 + 512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU()
        )
        self.conv3_d = nn.Sequential( 
            nn.ConvTranspose3d(in_channels=256 + 256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU()
        )
        self.conv4_d = nn.Sequential( 
            nn.ConvTranspose3d(in_channels=128 + 128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU()
        )
        self.final_d = nn.Sequential( 
            nn.ConvTranspose3d(in_channels=64 + 64, out_channels=image_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Tanh() # 数据归一化到 [-1, 1]，所以使用 Tanh
        )
    
    # === 移除 reparameterize 函数，因为它不再需要 ===
    # def reparameterize(self, mu, logvar):
    #     logvar = torch.clamp(logvar, min=-5.0, max=5.0)
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.rand_like(std)
    #     z = eps.mul(std).add_(mu)
    #     return z

    # === 修改 bottleneck 函数，直接输出 z ===
    def bottleneck(self, h):
        # mu, logvar = self.fc1(h), self.fc2(h) # 原始 CVAE
        # z = self.reparameterize(mu, logvar) # 原始 CVAE
        z = self.fc_encode(h) # AE 直接输出潜在向量 z
        return z # AE 只返回 z

    # === 修改 encode 函数，返回 z 和 skip connections ===
    def encode(self, x):
        e1 = self.conv1_e(x) # (64, 16, 16, 16)
        e2 = self.conv2_e(e1) # (128, 8, 8, 8)
        e3 = self.conv3_e(e2) # (256, 4, 4, 4)
        e4 = self.conv4_e(e3) # (512, 2, 2, 2)
        e5 = self.conv5_e(e4) # (1024, 1, 1, 1) # Deepest encoder output

        h_flat = e5.view(e5.size(0), -1) # Flatten final encoder output (1024)
        z = self.bottleneck(h_flat) # 获取潜在向量 z
        return z, e1, e2, e3, e4, e5 # 返回 z 和所有用于跳跃连接的中间特征图

    # decode 函数保持不变，因为它是从 z_reconstructed 和 skips 进行解码
    def decode(self, z_reconstructed, e_skips):
        e1, e2, e3, e4, e5 = e_skips

        d0 = self.initial_d(z_reconstructed) 
        d1 = self.conv1_d(torch.cat([d0, e5], dim=1)) 
        d2 = self.conv2_d(torch.cat([d1, e4], dim=1)) 
        d3 = self.conv3_d(torch.cat([d2, e3], dim=1)) 
        d4 = self.conv4_d(torch.cat([d3, e2], dim=1)) 
        decoded_x = self.final_d(torch.cat([d4, e1], dim=1)) 

        return decoded_x

    # representation 函数可以保留，因为它只用于提取潜在表示，但不再是 mu
    def representation(self, x):
        z, _, _, _, _, _ = self.encode(x) # 仅获取 z
        return z

    # === 修改 forward 函数的返回值 ===
    def forward(self, x):
        z, e1, e2, e3, e4, e5 = self.encode(x) # encode 现在直接返回 z
        
        # z_representation 是 z 本身
        z_representation = z 

        # z 经过 fc3 变换后，形状与 encoder_final_flat_dim 相同，才能送入 UnFlatten
        z_reconstructed_from_latent_space = self.fc3(z) 

        decoded_x = self.decode(z_reconstructed_from_latent_space, (e1, e2, e3, e4, e5)) 
        
        # 对于 AE，不再返回 mu 和 logvar，而是返回重建结果，z，以及 z 作为表示
        return decoded_x, z_representation, torch.tensor(0.0), torch.tensor(0.0) # mu 和 logvar 设为 0.0


# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)

# class UnFlatten(nn.Module):
#     def __init__(self, target_shape): # target_shape is a tuple like (channels, D, H, W)
#         super(UnFlatten, self).__init__()
#         self.target_shape = target_shape
#     def forward(self, input):
#         return input.view(input.size(0), *self.target_shape)

# class CVAE_3D(nn.Module):
#     def __init__(self, image_channels=1, z_dim=1024):
#         super(CVAE_3D, self).__init__()
#         print()
#         print("[INFO] instantiating pytorch model: 3D CVAE (Symmetrical U-Net with Skip Connections)")

#         # Encoder (5 layers, stride=2, padding=1)
#         # Input: (C, 32, 32, 32)
#         self.conv1_e = nn.Sequential( # Output: (64, 16, 16, 16)
#             nn.Conv3d(in_channels=image_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=64),
#             nn.ReLU()
#         )
#         self.conv2_e = nn.Sequential( # Output: (128, 8, 8, 8)
#             nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=128),
#             nn.ReLU()
#         )
#         self.conv3_e = nn.Sequential( # Output: (256, 4, 4, 4)
#             nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=256),
#             nn.ReLU()
#         )
#         self.conv4_e = nn.Sequential( # Output: (512, 2, 2, 2)
#             nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=512),
#             nn.ReLU()
#         )
#         # Bottleneck Encoder part (final downsample to 1x1x1)
#         self.conv5_e = nn.Sequential( # Output: (1024, 1, 1, 1)
#             nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=1024),
#             nn.ReLU()
#         )

#         self.encoder_final_flat_dim = 1024 * 1 * 1 * 1 # Flatten 后的维度是 1024
#         self.fc1 = nn.Linear(self.encoder_final_flat_dim, z_dim)
#         self.fc2 = nn.Linear(self.encoder_final_flat_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, self.encoder_final_flat_dim)

#         # Decoder (5 layers ConvTranspose3d, stride=2, padding=1)
#         # Initial layer after FC3 + UnFlatten
#         self.initial_d = nn.Sequential(
#             UnFlatten(target_shape=(1024, 1, 1, 1)), # From FC3 to (1024, 1, 1, 1)
#             nn.BatchNorm3d(num_features=1024),
#             nn.ReLU()
#         )
        
#         # Layer 1: Takes (decoder_input + skip_from_e5)
#         # Decoded output will be (512, 2, 2, 2)
#         self.conv1_d = nn.Sequential( 
#             nn.ConvTranspose3d(in_channels=1024 + 1024, out_channels=512, kernel_size=4, stride=2, padding=1, output_padding=0),
#             nn.BatchNorm3d(num_features=512),
#             nn.ReLU()
#         )
#         # Layer 2: Takes (decoder_input + skip_from_e4)
#         # Decoded output will be (256, 4, 4, 4)
#         self.conv2_d = nn.Sequential( 
#             nn.ConvTranspose3d(in_channels=512 + 512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0),
#             nn.BatchNorm3d(num_features=256),
#             nn.ReLU()
#         )
#         # Layer 3: Takes (decoder_input + skip_from_e3)
#         # Decoded output will be (128, 8, 8, 8)
#         self.conv3_d = nn.Sequential( 
#             nn.ConvTranspose3d(in_channels=256 + 256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0),
#             nn.BatchNorm3d(num_features=128),
#             nn.ReLU()
#         )
#         # Layer 4: Takes (decoder_input + skip_from_e2)
#         # Decoded output will be (64, 16, 16, 16)
#         self.conv4_d = nn.Sequential( 
#             nn.ConvTranspose3d(in_channels=128 + 128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0),
#             nn.BatchNorm3d(num_features=64),
#             nn.ReLU()
#         )
#         # Final Layer (output layer): Takes (decoder_input + skip_from_e1)
#         # Decoded output will be (image_channels, 32, 32, 32)
#         self.final_d = nn.Sequential( 
#             nn.ConvTranspose3d(in_channels=64 + 64, out_channels=image_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
#             nn.Tanh() # Data normalized to [-1, 1], so use Tanh
#         )

#     def reparameterize(self, mu, logvar):
#         logvar = torch.clamp(logvar, min=-5.0, max=5.0) # Clip logvar for numerical stability
#         std = torch.exp(0.5 * logvar)
#         eps = torch.rand_like(std)
#         z = eps.mul(std).add_(mu)
#         return z

#     def bottleneck(self, h):
#         mu, logvar = self.fc1(h), self.fc2(h)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar

#     # 编码器前向传播并返回最终的展平特征和所有中间特征图以用于跳跃连接
#     def encode(self, x):
#         e1 = self.conv1_e(x) # (64, 16, 16, 16)
#         e2 = self.conv2_e(e1) # (128, 8, 8, 8)
#         e3 = self.conv3_e(e2) # (256, 4, 4, 4)
#         e4 = self.conv4_e(e3) # (512, 2, 2, 2)
#         e5 = self.conv5_e(e4) # (1024, 1, 1, 1) # Deepest encoder output

#         h_flat = e5.view(e5.size(0), -1) # Flatten final encoder output (1024)
#         # 返回展平特征 h_flat 和所有用于跳跃连接的中间特征图
#         return h_flat, e1, e2, e3, e4, e5 

#     def decode(self, z_reconstructed, e_skips):
#         # e_skips 包含了 (e1, e2, e3, e4, e5)
#         e1, e2, e3, e4, e5 = e_skips

#         # 解码器从潜在空间映射后的特征开始
#         d0 = self.initial_d(z_reconstructed) # From FC3 to (1024, 1, 1, 1)

#         # 逐层上采样并拼接跳跃连接 (从编码器深层到浅层)
#         # ConvTranspose output spatial size: (input_dim - 1)*stride - 2*padding + kernel_size + output_padding
#         # For stride=2, kernel_size=4, padding=1, output_padding=0 -> (input_dim-1)*2 - 2 + 4 = 2*input_dim - 2 + 2 = 2*input_dim
        
#         d1 = self.conv1_d(torch.cat([d0, e5], dim=1)) # d0 (1024,1,1,1) + e5 (1024,1,1,1) -> (2048,1,1,1) -> ConvT -> (512,2,2,2)
#         d2 = self.conv2_d(torch.cat([d1, e4], dim=1)) # d1 (512,2,2,2) + e4 (512,2,2,2) -> (1024,2,2,2) -> ConvT -> (256,4,4,4)
#         d3 = self.conv3_d(torch.cat([d2, e3], dim=1)) # d2 (256,4,4,4) + e3 (256,4,4,4) -> (512,4,4,4) -> ConvT -> (128,8,8,8)
#         d4 = self.conv4_d(torch.cat([d3, e2], dim=1)) # d3 (128,8,8,8) + e2 (128,8,8,8) -> (256,8,8,8) -> ConvT -> (64,16,16,16)
        
#         # 最后一层：d4 (64,16,16,16) + e1 (64,16,16,16) -> (128,16,16,16) -> ConvT -> (C,32,32,32)
#         decoded_x = self.final_d(torch.cat([d4, e1], dim=1)) 

#         return decoded_x

#     def forward(self, x):
#         h_flat, e1, e2, e3, e4, e5 = self.encode(x) # 编码器输出 h_flat 和所有中间特征图

#         z, mu, logvar = self.bottleneck(h_flat) # bottleneck 操作在 h_flat 上
#         z_reconstructed_from_latent_space = self.fc3(z) # 将 z 映射回 Flatten 前的维度

#         # 将所有跳跃连接的特征图打包传递给 decode
#         decoded_x = self.decode(z_reconstructed_from_latent_space, (e1, e2, e3, e4, e5)) 
        
#         return decoded_x, mu, logvar, mu # 返回 mu 作为 z_representation



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
