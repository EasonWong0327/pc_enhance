import os
import torch
import re
import open3d as o3d  # 用于读取 PLY 文件
import numpy as np
import MinkowskiEngine as ME
import subprocess
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import shutil
from collections import defaultdict
import time
device = torch.device('cuda:0')


class MyDeepNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MyDeepNet, self).__init__()

        # 第一层卷积块
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, 32, kernel_size=3, dimension=3
        )
        self.norm1 = ME.MinkowskiBatchNorm(32)
        self.relu1 = ME.MinkowskiReLU()

        # 第一组卷积块
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )

        # 第二组卷积块
        self.block2 = nn.Sequential(
            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )

        # 第三组卷积块
        self.block3 = nn.Sequential(
            ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU(),
        )

        # 第四组卷积块
        self.block4 = nn.Sequential(
            ME.MinkowskiConvolution(256, 512, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )

        # 加入第五组卷积块（加深网络结构）
        self.block5 = nn.Sequential(
            ME.MinkowskiConvolution(512, 1024, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(1024),
            ME.MinkowskiReLU(),
        )

        # 1x1卷积用于调整形状
        self.adjust1 = ME.MinkowskiConvolution(32, 64, kernel_size=1, dimension=3)
        self.adjust2 = ME.MinkowskiConvolution(64, 128, kernel_size=1, dimension=3)
        self.adjust3 = ME.MinkowskiConvolution(128, 256, kernel_size=1, dimension=3)
        self.adjust4 = ME.MinkowskiConvolution(256, 512, kernel_size=1, dimension=3)
        self.adjust5 = ME.MinkowskiConvolution(512, 1024, kernel_size=1, dimension=3)

        # 最后的卷积层输出
        self.conv_out = ME.MinkowskiConvolution(1024, out_channels, kernel_size=1, dimension=3)

    def forward(self, x):
        # 第一层卷积
        x = self.relu1(self.norm1(self.conv1(x)))

        # 第一组卷积块
        residual = self.adjust1(x)  # 调整 residual 的形状
        x = self.block1(x)
        x += residual  # 残差连接

        # 第二组卷积块
        residual = self.adjust2(x)  # 调整 residual 的形状
        x = self.block2(x)
        x += residual  # 残差连接

        # 第三组卷积块
        residual = self.adjust3(x)  # 调整 residual 的形状
        x = self.block3(x)
        x += residual  # 残差连接

        # 第四组卷积块
        residual = self.adjust4(x)  # 调整 residual 的形状
        x = self.block4(x)
        x += residual  # 残差连接

        # 第五组卷积块
        residual = self.adjust5(x)  # 调整 residual 的形状
        x = self.block5(x)
        x += residual  # 残差连接

        # 最后的卷积输出
        x = self.conv_out(x)
        
        return x


def load_ply(file_path):
    """读取 PLY 文件并返回 (N, 6) 的点云数组"""
    print(f"正在加载文件：{file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points


def save_ply_with_open3d(points, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
    print(f"保存点云到文件 {save_path}")


def load_model_for_prediction(model_class, model_path, device=device):
    """
    加载模型权重并将模型移动到指定设备上。

    Args:
        model_class: 模型类。
        model_path: 模型权重文件的路径。
        device: 设备（'cuda' 或 'cpu'）。

    Returns:
        已加载权重并移动到指定设备的模型。
    """
    model = model_class().to(device)  # 初始化模型并移动到指定设备
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型权重
    model.eval()  # 设置模型为评估模式
    return model


def normalize_coordinates(coords, global_min, global_max):
    """
    将坐标归一化到 [0, 1] 范围内。
    """
    # 确保所有张量都在同一个设备上
    device = coords.device
    global_min = global_min.to(device)
    global_max = global_max.to(device)

    # 进行归一化操作
    normalized_coords = (coords - global_min) / (global_max - global_min)
    return normalized_coords


def denormalize_coordinates(coords, global_min, global_max):
    """
    将归一化的坐标还原到原始范围。
    """
    device = coords.device
    global_min = global_min.to(device)
    global_max = global_max.to(device)

    denormalized_coords = coords * (global_max - global_min) + global_min
    denormalized_coords = denormalized_coords.int()
    return denormalized_coords


def predict_on_blocks(model, blocks, global_min, global_max):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型放在GPU上

    global_min = torch.tensor(global_min, dtype=torch.float32, device=device)
    global_max = torch.tensor(global_max, dtype=torch.float32, device=device)
    
    coords_B = torch.tensor(blocks[:, :3], dtype=torch.float32).to(device)
    feats_B = torch.tensor(blocks[:, 3:], dtype=torch.float32).to(device)

    normalized_coords_B = normalize_coordinates(coords_B, global_min, global_max)

    # 添加第四维用于批次索引
    batch_index = torch.full((coords_B.shape[0], 1), 0, dtype=torch.int32, device=device)
    int_coords_B = torch.cat([batch_index, coords_B.int()], dim=1)

    inputs = ME.SparseTensor(features=normalized_coords_B, coordinates=int_coords_B)
#     print(inputs)
#     time.sleep(100000)
    output = model(inputs)
    print(output.F)
    denormalize_output = denormalize_coordinates(output.F, global_min, global_max)
    print(denormalize_output)

    print('shape check:',feats_B.shape, inputs.shape,output.shape)
    # if output.F.shape[0] != normalized_coords_B.shape[0]:
    #     print(f"Warning: Mismatch in tensor sizes for block {block_idx}. Skipping this block.")
    #     continue  # 跳过点数不一致的块，避免错误
    # print(normalized_coords_B)
#     print(inputs)
#     print(output.F)
    predicted_coords = normalized_coords_B + output.F

    denormalize_coords = denormalize_coordinates(predicted_coords, global_min, global_max)
    # 使用 detach() 去除梯度信息
    predicted_points = torch.cat((denormalize_coords, feats_B), dim=1).detach().cpu().numpy()

    # 清理未使用的张量
    del coords_B, feats_B, normalized_coords_B, inputs, output, predicted_coords, denormalize_coords
    torch.cuda.empty_cache()

    return predicted_points


def _load_ply(file_path):
    # 只要坐标
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def _save_ply(points, file_path):
    # 使用open3d保存点云数据为ply格式，不包含颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 只保存坐标信息
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

def merge_blocks(block_folder, output_folder):
    """
    根据文件名中的特定部分合并块并保存为多个完整的PLY文件。
    """
    blocks = defaultdict(list)

    # 遍历块文件夹中的所有块文件
    for block_file in sorted(os.listdir(block_folder)):
        if block_file.endswith('.ply'):
            # 提取文件名中的特定部分
            parts = block_file.split('_')
            if len(parts) > 3:
                key = '_'.join(parts[:3])
            else:
                key = '_'.join(parts[:2])
            block_path = os.path.join(block_folder, block_file)
            blocks[key].append(block_path)

    # 合并每个特定部分的块
    for key, block_files in blocks.items():
        all_points = []

        for block_file in block_files:
            block_points = _load_ply(block_file)
            all_points.append(block_points)

        # 合并所有块
        all_points = np.vstack(all_points)

        # 确定输出文件名
        if 'vox10' in key:
            output_file = os.path.join(output_folder, f"{key}_origin.ply")
        elif 'S26C03R03_rec' in key:
            output_file = os.path.join(output_folder, f"{key}.ply")

        # 保存为一个完整的PLY文件
        _save_ply(all_points, output_file)
        print(f"合并后的点云保存为: {output_file}")


def main(compress_dir, model_path, output_dir,global_min, global_max):

    try:
        model = load_model_for_prediction(MyDeepNet, model_path, device)
    except RuntimeError as e:
        print(str(e))

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 获取 compress_dir 中的所有 .ply 文件
    files_B = sorted([os.path.join(compress_dir, f) for f in os.listdir(compress_dir) if f.endswith('.ply')])

    for file_B in files_B:
        # 提取文件名部分，用于输出文件的命名
        base_name_B = os.path.basename(file_B)

        # 加载并切分文件 B
        points_B = load_ply(file_B)

        # 预测并合并所有块
        predicted_points = predict_on_blocks(model, points_B, global_min, global_max)

        # 保存合并后的点云
        base_name_B = base_name_B.replace('.ply','')
        output_file_path = os.path.join(output_dir, f"{base_name_B}_predicted.ply")
        save_ply_with_open3d(predicted_points, output_file_path)


if __name__ == "__main__":
    compress_dir = './data30/soldier/block/adjusted_chunks_B'
    global_min = np.array([0.0, 0.0, 0.0])
    global_max = np.array([620.0, 1024.0, 662.0])

    model_path = './model/my_net_residual_1117/200_model_residual.pth'
    output_dir = './test30/soldier/predict_my_net_residual_1117_model_200_single'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # 调用 main 函数进行预测
    main(compress_dir, model_path, output_dir,global_min=global_min, global_max=global_max)
    
    merged_output_dir = './test30/soldier/predict_my_net_residual_1117_model_200_merged'
    if os.path.exists(merged_output_dir):
        shutil.rmtree(merged_output_dir)
    os.makedirs(merged_output_dir, exist_ok=True)
    merge_blocks(block_folder=output_dir, output_folder=merged_output_dir)