import numpy as np
import os, sys
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import pdb
from skimage.transform import resize
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd

class config:
    dir = "./100"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = 200
    image_size = 32
    beta_start = 0.0001
    beta_end = 0.02
    list_num = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 31, 32, 33, 34, 35]
    list_Folder = [43, 55, 64, 73, 85, 97, 103, 112, 121, 127, 136, 145, 154, 163, 169, 178, 188, 202, 213, 225]
    list_time_steps = [1, 2, 3, 4, 5]

class CFD3DDataset(Dataset):
    def __init__(self, img_dir, list_num=None, list_Folder=None, list_time_steps=None, global_min_val=None, global_max_val=None):
        self.dir = img_dir # 例如: "../100"
        self.folder_ids = list_Folder if list_Folder is not None else config.list_Folder
        self.num_ids = list_num if list_num is not None else config.list_num
        self.time_step_ids = list_time_steps if list_time_steps is not None else config.list_time_steps
        
        # 预先构建所有样本的唯一标识符列表 (folder_id, time_step_id)
        # 每个 (folder_id, time_step_id) 对应一个 32x32x32 的完整 CKM 样本
        self.sample_identifiers = []
        for folder_id in self.folder_ids:
            for time_step_id in self.time_step_ids:
                self.sample_identifiers.append((folder_id, time_step_id))
        
        self.data_len = len(self.sample_identifiers) # 你的总样本数 (20 * 5 = 100)

        # === 核心修正 1: 全局归一化参数的设置和计算 ===
        # 只有当 global_min_val 或 global_max_val 未提供时才计算
        self.global_min_val = global_min_val
        self.global_max_val = global_max_val

        if self.global_min_val is None or self.global_max_val is None:
            print("[INFO] Computing global min/max for normalization. This might take some time...")
            all_values = []
            # 遍历所有样本来计算全局统计量
            for folder_id, time_step_id in self.sample_identifiers:
                for num_id_z in self.num_ids:
                    # 构建每个 .p2m 文件的路径
                    file_path = os.path.join(self.dir, str(folder_id), 
                                             f"Urban.pg.t{time_step_id:03d}_{folder_id}.r{num_id_z:03d}.p2m")
                    if not os.path.exists(file_path):
                        print(f"Error: Required file not found for global min/max calculation: {file_path}")
                        # 可以在这里选择跳过或引发错误
                        continue 
                        
                    df = pd.read_csv(file_path, sep=' ', skiprows=2)
                    matrix = df.pivot(index='<Y(m)>', columns='<X(m)>', values='<PathGain(dB)>').values
                    all_values.extend(matrix.flatten().tolist()) # 收集所有值

            if not all_values: # 防止所有文件都缺失导致all_values为空
                 raise RuntimeError("No data values collected for global min/max calculation. Check file paths and data loading.")

            self.global_min_val = np.min(all_values)
            self.global_max_val = np.max(all_values)
            print(f"[INFO] Global min/max computed: [{self.global_min_val}, {self.global_max_val}]")
        else:
            print(f"[INFO] Using provided global normalization range: [{self.global_min_val}, {self.global_max_val}]")

        print(f"[INFO] ChannelDataset initialized with {self.data_len} samples.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # 根据索引获取当前的样本标识符 (folder_id, time_step_id)
        folder_id, time_step_id = self.sample_identifiers[idx]
        matrices = []  # 存储构成 3D 立方体的所有 32x32 的 2D 矩阵 (32 层)

        for num_id_z in self.num_ids: # 遍历 32 层
            # 构建每个 .p2m 文件的路径
            file_path = os.path.join(self.dir, str(folder_id), 
                                     f"Urban.pg.t{time_step_id:03d}_{folder_id}.r{num_id_z:03d}.p2m")
            
            # 使用 pd.read_csv 读取数据
            df = pd.read_csv(file_path, sep=' ', skiprows=2)
            
            # 使用 pivot 将数据转换为 32x32 矩阵
            matrix = df.pivot(index='<Y(m)>', columns='<X(m)>', values='<PathGain(dB)>').values
            matrices.append(matrix)
        
        # 将列表转换为 NumPy 数组，形状为 (D, H, W) -> (32, 32, 32)
        data_np = np.array(matrices, dtype=np.float32) 

        # === 核心修正 2: 应用全局归一化 ===
        if self.global_max_val - self.global_min_val == 0:
            data_normalized = np.zeros_like(data_np)
        else:
            data_normalized = (data_np - self.global_min_val) / (self.global_max_val - self.global_min_val)
            data_normalized=data_normalized
        
        # 将数据从 (D, H, W) 形状转换为 (C, D, H, W) -> (1, 32, 32, 32)
        data_tensor = torch.tensor(data_normalized, dtype=torch.float32).unsqueeze(0) 

        # 如果你希望模型输入是 [-1, 1] 范围，则在此处进行线性变换：
        data_tensor = data_tensor * 2.0 - 1.0 # 将 [0,1] 变为 [-1,1]
        
        return data_tensor # 返回形状为 (1, 32, 32, 32) 的 Tensor
    

    # def __init__(self, data_directory, no_simulations, simulation_timesteps, transforms=None):
    #     """
    #     data_directory: path to directory that contains subfolders with the npy files
    #     Subfolders are folders containing each component of velocity: extract_cubes_U0_reduced
    #     """

    #     print()
    #     print("[INFO] started instantiating 3D CFD pytorch dataset")

    #     self.data_directory = data_directory
    #     self.no_simulations = no_simulations # 96
    #     self.simulation_timesteps = simulation_timesteps # 100
    #     self.transforms = transforms

    #     # data_dir = "../cfd_data/HVAC_DUCT/cubes/coords_3d/"
    #     data_directory_U0 = self.data_directory + "extract_cubes_U0_reduced/"
    #     data_directory_U1 = self.data_directory + "extract_cubes_U1_reduced/"
    #     data_directory_U2 = self.data_directory + "extract_cubes_U2_reduced/"

    #     # read cubes data from directories
    #     cubes_U0_dict = self._load_3D_cubes(data_directory_U0)
    #     cubes_U1_dict = self._load_3D_cubes(data_directory_U1)
    #     cubes_U2_dict = self._load_3D_cubes(data_directory_U2)

    #     # compare all folders have same simulation parameters
    #     if self._compare_U_sim_keys(cubes_U0_dict, cubes_U1_dict) and \
    #        self._compare_U_sim_keys(cubes_U0_dict, cubes_U2_dict) and \
    #        self._compare_U_sim_keys(cubes_U1_dict, cubes_U2_dict):
    #         print("[INFO] all folders have same keys (simulations)")
    #     else:
    #         print("[INFO] the folders don't have the same keys (simulations)")
    #         quit()

    #     # concatenate all velocity components into one dictionary data structure
    #     cubes_U_all_dict = self._merge_velocity_components_into_dict(cubes_U0_dict, cubes_U1_dict, cubes_U2_dict)

    #     # creates a list of length timesteps x simulations, each element is a numpy array with cubes size (21,21,21,3)
    #     # cubes_U_all_channels: 9600 with shape (21,21,21,3)
    #     self.cubes_U_all_channels = self._concatenate_3_velocity_components(cubes_U_all_dict)
    #     print("[INFO] cubes dataset length:", len(self.cubes_U_all_channels))
    #     print("[INFO] single cube shape:", self.cubes_U_all_channels[0].shape)
    #     self.data_len = len(self.cubes_U_all_channels)

    #     # stack all cubes in a final numpy array numpy (9600, 21, 21, 21, 3)
    #     self.stacked_cubes = np.stack(self.cubes_U_all_channels, 0)

    #     print()
    #     print("[INFO] mean and std of the cubes dataset along 3 channels")
    #     # note: not using mean and std separately, just calling them in standardize function (below)
    #     # note: only standardize data to mean 0 and std 1
    #     self.mean, self.std = self._compute_mean_std_dataset(self.stacked_cubes)
    #     print("mean:", self.mean)
    #     print("std:", self.std)

    #     # standardize data from here
    #     print()
    #     print("[INFO] standardize data to mean 0 and std 1")
    #     self.standardized_cubes = self._standardize_cubes(self.stacked_cubes)
    #     print("mean after standardization:", self.standardized_cubes.mean(axis=(0,1,2,3)))
    #     print("std after standardization:", self.standardized_cubes.std(axis=(0,1,2,3)))

    #     print()
    #     print("[INFO] finished instantiating 3D CFD pytorch dataset")

    # def _load_3D_cubes(self, data_directory):
    #     """
    #     Saves 3D CFD data in a dictionary.
    #     Keys correspond to .npy file name
    #     Values correspond to arrays of size (21, 21, 21, 100)
    #     """

    #     cubes = {}

    #     for filename in os.listdir(data_directory):
    #         if filename.endswith(".npy"):
    #             # set key without Ui character (for later key matching)
    #             cubes[filename[2:]] = (np.load(data_directory + "/" + filename))

    #     return cubes

    # def _compare_U_sim_keys(self, cube1, cube2):
    #     """
    #     Asserts that two folders with two different velocity componentes
    #     have same simulation parameters (based on npy file name)
    #     """
    #     matched_keys = 0
    #     for key in cube1:
    #         if key in cube2:
    #             matched_keys += 1

    #     if matched_keys == self.no_simulations:
    #         return True
    #     else:
    #         return False

    # def _merge_velocity_components_into_dict(self, cubes_U0, cubes_U1, cubes_U2):
    #     """
    #     Concatenates all velocity components U0, U1, U2  based on
    #     key (simulation name) into a dictionary data structure.
    #     """
    #     cubes_U = defaultdict(list)

    #     for d in (cubes_U0, cubes_U1, cubes_U2): # you can list as many input dicts as you want here
    #         for key, value in d.items():
    #             cubes_U[key].append(value)

    #     # this returns a list of sublists, each sublists contains 3 arrays (corresponding to U0, U1, U2)
    #     print("[INFO] velocity components concatenated into list")
    #     return cubes_U

    # def _concatenate_3_velocity_components(self, cubes_dict):
    #     """
    #     """
    #     cubes_3_channels = []

    #     for key, value in cubes_dict.items():
    #         # split temporal dependency of simulations
    #         for timestep in range(0, self.simulation_timesteps):
    #             # fetch velocity compponents
    #             U0 = cubes_dict[key][0][:,:,:,timestep] # one cube, three channels, one time step
    #             U1 = cubes_dict[key][1][:,:,:,timestep]
    #             U2 = cubes_dict[key][2][:,:,:,timestep]

    #             # concatenate as channels (21, 21, 21, 3)
    #             U_all_channels = np.concatenate((U0[...,np.newaxis],
    #                                              U1[...,np.newaxis],
    #                                              U2[...,np.newaxis]),
    #                                              axis=3)

    #             cubes_3_channels.append(U_all_channels)

    #     return cubes_3_channels

    # def _compute_mean_std_dataset(self, data):
    #     """
    #     Gets mean and standard deviation values for 3 channels of 3D cube data set.
    #     It computes mean and standard deviation of full dataset (not on batches)

    #     Based on: https://stackoverflow.com/questions/47124143/mean-value-of-each-channel-of-several-images
    #     Based on: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6

    #     Returns 1D arrays for mean and std for corresponding channels.
    #     """

    #     mean = 0.
    #     std = 0.

    #     mean = np.mean(data, axis=(0,1,2,3), dtype=np.float64) # axis=(0,1,2,3)
    #     std = np.std(data, axis=(0,1,2,3), dtype=np.float64)

    #     return torch.from_numpy(mean), torch.from_numpy(std)

    # def _standardize_cubes(self, data):
    #     """
    #     Performs standard normalization on given array.
    #     """
    #     # (9600, 21, 21, 21, 3)
    #     # means = [7.5, 6.3, 1.2]
    #     return (data - data.mean(axis=(0,1,2,3), keepdims=True)) / data.std(axis=(0,1,2,3), keepdims=True)

    # def __getitem__(self, index):
    #     """
    #     Returns a tensor cube of shape (3,21,21,21) normalized by
    #     substracting mean and dividing std of dataset computed beforehand.
    #     """

    #     single_cube_numpy = self.standardized_cubes[index] # (21, 21, 21, 3)

    #     # min-max normalization, clipping and resizing
    #     single_cube_minmax = self._minmax_normalization(single_cube_numpy) # (custom function)
    #     single_cube_transformed = np.clip(self._scale_by(np.clip(single_cube_minmax-0.1, 0, 1)**0.4, 2)-0.1, 0, 1) # (from tutorial)
    #     single_cube_resized = resize(single_cube_transformed, (21, 21, 21), mode='constant') # (21,21,21)

    #     # swap axes from numpy shape (21, 21, 21, 3) to torch shape (3, 21, 21, 21) this is for input to Conv3D
    #     # single_cube_reshaped = np.transpose(single_cube_minmax, (3, 1, 2, 0))
    #     single_cube_reshaped = np.transpose(single_cube_resized, (3, 1, 2, 0))

    #     # convert cube to torch tensor
    #     single_cube_tensor = torch.from_numpy(single_cube_reshaped)

    #     # NOTE: not applying ToTensor() because it only works with 2D images
    #     # if self.transforms is not None:
    #         # single_cube_tensor = self.transforms(single_cube_normalized)
    #         # single_cube_tensor = self.transforms(single_cube_PIL)

    #     return single_cube_tensor

    # def _minmax_normalization(self, data):
    #    """
    #    Performs MinMax normalization on given array. Range [0, 1]
    #    """

    #    # data shape (21, 21, 21, 3)
    #    data_min = np.min(data, axis=(0,1,2))
    #    data_max = np.max(data, axis=(0,1,2))

    #    return (data-data_min)/(data_max - data_min)

    # def _scale_by(self, arr, fac):
    #     mean = np.mean(arr)
    #     return (arr-mean)*fac + mean

    # def __len__(self):
    #     return self.data_len
