# 把kd-tree拆出来

import os
import re  # 用于从文件名中提取编号
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d  # 用于读取 PLY 文件
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from MinkowskiEngine import utils as ME_utils
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import logging
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用第二块GPU



# 全局最小值和最大值
global_min = np.array([0.0, 0.0, 0.0])
global_max = np.array([620.0, 1024.0, 662.0])


def normalize_coordinates(coords, global_min, global_max):
    """
    将坐标归一化到 [0, 1] 范围内。
    """
    return (coords - global_min) / (global_max - global_min)

def remove_duplicates(points):
    """
    去除点云数据中的重复点，仅考虑坐标部分。

    参数:
    points (numpy.ndarray): 点云数据，形状为 (N, 6)，前3列为坐标，后3列为色彩信息。

    返回:
    numpy.ndarray: 去除重复点后的点云数据。
    """
    # 提取坐标部分
    coordinates = points[:, :3]

    # 获取唯一坐标及其索引
    unique_coords, indices = np.unique(coordinates, axis=0, return_index=True)

    # 根据索引获取对应的色彩信息
    unique_points = points[indices]

    return unique_points

def has_duplicates(points, tol=1e-9):
    """
    检查点云数据集中是否存在重复的点。

    参数:
    points (torch.Tensor or numpy.ndarray): 点云数据，形状为 (N, D)。
    tol (float): 判断重复点的容差，默认为 1e-9。

    返回:
    bool: 如果存在重复的点，则返回 True；否则返回 False。
    """
    if isinstance(points, torch.Tensor):
        # 如果是 GPU 张量，先移动到 CPU
        if points.is_cuda:
            points = points.cpu()
        # 转换为 NumPy 数组
        points = points.numpy()

    tree = KDTree(points)
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=2)
        if distances[1] < tol:
            return True
    return False


def has_duplicates_output(points, tol=1e-9):
    """
    debug用
    检查点云数据集中是否存在重复的点，并输出重复的坐标。

    参数:
    points (torch.Tensor or numpy.ndarray): 点云数据，形状为 (N, D)。
    tol (float): 判断重复点的容差，默认为 1e-9。

    返回:
    tuple: (bool, list)，如果存在重复的点，则返回 (True, 重复点的列表)；否则返回 (False, 空列表)。
    """
    if isinstance(points, torch.Tensor):
        # 如果是 GPU 张量，先移动到 CPU
        if points.is_cuda:
            points = points.cpu()
        # 转换为 NumPy 数组
        points = points.numpy()

    tree = KDTree(points)
    duplicates = []
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=2)
        if distances[1] < tol:
            duplicates.append(point)

    has_dup = len(duplicates) > 0
    return has_dup, duplicates


def chunk_point_cloud_fixed_size(points, block_size=256, cube_size=1024, min_points_ratio=0.1, device='cuda'):
    """
    将点云数据切分为固定大小的块，支持可选偏移，并确保块的数量和大小一致。
    """
    points = torch.tensor(points, device=device, dtype=torch.float32)  # 使用 float32
    coords = points[:, :3]

    min_points_threshold = int(block_size ** 3 * min_points_ratio)

    x_range = torch.arange(0, cube_size, block_size, device=device)
    y_range = torch.arange(0, cube_size, block_size, device=device)
    z_range = torch.arange(0, cube_size, block_size, device=device)

    blocks = []

    for x in x_range:
        for y in y_range:
            for z in z_range:
                mask = (
                    (coords[:, 0] >= x) & (coords[:, 0] < x + block_size) &
                    (coords[:, 1] >= y) & (coords[:, 1] < y + block_size) &
                    (coords[:, 2] >= z) & (coords[:, 2] < z + block_size)
                )

                block_coords = coords[mask]
                if len(block_coords) >= min_points_threshold:
                    blocks.append((block_coords.cpu().numpy(), (x.item(), y.item(), z.item())))

                # 清理未使用的张量
                del block_coords
                torch.cuda.empty_cache()
    print(f"总切块数: {len(blocks)}")  # 添加打印信息
    return blocks


def adjust_points(chunk_A, chunk_B):
    """
    优化精简版
    调整chunk_A中的点以匹配chunk_B中的点，并确保没有重复点。

    参数：
        chunk_A (np.ndarray): chunk A中的点数组 (N x 3或N x 6，具体取决于维度)。
        chunk_B (np.ndarray): chunk B中的点数组 (M x 3或M x 6)。

    返回：
        np.ndarray: 调整后的chunk_A，包含与chunk_B最接近的点，数量与chunk_B相同。
    """
    if len(chunk_A) == 0 or len(chunk_B) == 0:
        return np.empty((0, chunk_B.shape[1]))  # 返回空数组，保持维度一致

    n_points = len(chunk_B)  # 设置n_points为chunk_B的长度
    tree = KDTree(chunk_A)
    adjusted_indices = set()  # 用于跟踪在chunk_A中唯一索引的集合
    adjusted_chunk_A = np.empty((n_points, chunk_A.shape[1]))  # 初始化调整后的chunk_A

    # 遍历chunk_B中的点，寻找在chunk_A中最近的唯一点
    for i, point in enumerate(chunk_B):
        distances, indices = tree.query(point, k=len(chunk_A))  # 获取chunk_A中所有点的排序距离

        # 寻找最近的未使用点
        for index in indices:
            if index not in adjusted_indices:
                adjusted_indices.add(index)
                adjusted_chunk_A[i] = chunk_A[index]
                break

    return adjusted_chunk_A


class PointCloudDataset(Dataset):
    def __init__(self, folder_A, folder_B, block_size, cube_size, min_points_ratio, output_folder, preprocess=True, mode=''):
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.block_size = block_size
        self.cube_size = cube_size
        self.min_points_ratio = min_points_ratio
        self.output_folder = output_folder
        self.preprocess = preprocess
        self.file_pairs = self._get_file_pairs()
        self.mode = mode

        # 数据预处理阶段执行kd-tree搜索并保存
        if preprocess:
            self._preprocess_data()

    def _get_file_pairs(self):
        # 获取folder_A和folder_B中的文件对
        files_A = sorted([f for f in os.listdir(self.folder_A) if f.endswith('.ply')])
        files_B = sorted([f for f in os.listdir(self.folder_B) if f.endswith('.ply')])
        return list(zip(files_A, files_B))

    def _preprocess_data(self):
        # 创建保存adjusted_chunks_A和adjusted_chunks_B的文件夹
        adjusted_A_folder = os.path.join(self.output_folder, 'adjusted_chunks_A')
        adjusted_B_folder = os.path.join(self.output_folder, 'adjusted_chunks_B')
        os.makedirs(adjusted_A_folder, exist_ok=True)
        os.makedirs(adjusted_B_folder, exist_ok=True)

        # 遍历文件对并进行kd-tree匹配
        for file_A, file_B in self.file_pairs:
            # 加载并处理点云文件
            print('开始处理：',self.folder_A, file_A,file_B)
            points_A = self._load_ply(os.path.join(self.folder_A, file_A))
            points_B = self._load_ply(os.path.join(self.folder_B, file_B))

            # points_A = self.load_ply(file_A)
            # points_B = self.load_ply(file_B)

            # 1.数据清洗：去除重复坐标   压缩后的ply有坐标重复
            check_compress = has_duplicates(points_B)
            if check_compress:
                points_B = remove_duplicates(points_B)
                # debug用
                # has_dup, duplicates = has_duplicates_output(points_B)
                # print(duplicates)
            # 2.数据切块
            chunks_A = chunk_point_cloud_fixed_size(points_A, self.block_size, self.cube_size, self.min_points_ratio)
            chunks_B = chunk_point_cloud_fixed_size(points_B, self.block_size, self.cube_size, self.min_points_ratio)

            if self.mode == 'test_process':
                chunks_A = chunks_A[0:8]
                chunks_B = chunks_B[0:8]

            # 3.kd-tree匹配
            adjusted_chunks_A = []
            adjusted_chunks_B = []
            for (chunk_A, index_A), (chunk_B, index_B) in zip(chunks_A, chunks_B):
                if index_A == index_B:
                    print('初始Chunk_B是否有重复：', has_duplicates(chunk_B))
                    # 根据chunk的大小调整A或B
                    if len(chunk_A) < len(chunk_B):
                        adjusted_chunk_B = adjust_points(chunk_B, chunk_A)  # 调整B以匹配A
                        print(
                            f"调整前的 chunk_B 点数: {chunk_B.shape[0]}，调整后的点数: {adjusted_chunk_B.shape[0]}，chunk_A的点数: {chunk_A.shape[0]}")
                        adjusted_chunks_B.append(adjusted_chunk_B)
                        adjusted_chunks_A.append(chunk_A)
                    else:
                        start_time = time.time()
                        adjusted_chunk_A = adjust_points(chunk_A, chunk_B)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"KD-Tree耗时: {elapsed_time:.4f} 秒")
                        print(
                            f"调整前的 chunk_A 点数: {chunk_A.shape[0]}，调整后的点数: {adjusted_chunk_A.shape[0]}，chunk_B的点数: {chunk_B.shape[0]}")
                        adjusted_chunks_A.append(adjusted_chunk_A)
                        adjusted_chunks_B.append(chunk_B)

            print('-------------------------------开始打印每块的点数------------------------')
            for i in range(len(adjusted_chunks_A)):
                print(f'第{i}块: ', adjusted_chunks_A[i].shape, adjusted_chunks_B[i].shape)

                # 保存adjusted_chunk_A和adjusted_chunk_B
                # 保存每一块的adjusted_chunk_A和adjusted_chunk_B
                file_A = file_A.replace('.ply','')
                file_B = file_B.replace('.ply','')
                self._save_ply(adjusted_chunks_A[i], os.path.join(adjusted_A_folder, f"{file_A}_block_{i}.ply"))
                self._save_ply(adjusted_chunks_B[i], os.path.join(adjusted_B_folder, f"{file_B}_block_{i}.ply"))

    def _load_ply(self, file_path):
        # 使用open3d加载ply文件
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # print(points.shape, colors.shape)
        if colors.shape[0] > 0 and colors.shape[1] > 0:  # 检查颜色信息是否存在
            return np.hstack((points, colors))
        else:
            return points

    def _save_ply(self, points, file_path):
        # 使用open3d保存点云数据为ply格式，不包含颜色
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 只保存坐标信息
        o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        # 加载已经保存的预处理结果
        file_A, file_B = self.file_pairs[idx]
        adjusted_chunk_A = self._load_ply(os.path.join(self.folder_A, file_A))
        adjusted_chunk_B = self._load_ply(os.path.join(self.folder_B, file_B))

        # 转换为张量并返回
        adjusted_chunk_A = torch.tensor(adjusted_chunk_A, dtype=torch.float32)
        adjusted_chunk_B = torch.tensor(adjusted_chunk_B, dtype=torch.float32)

        return adjusted_chunk_A, adjusted_chunk_B

def save_point_cloud_as_ply(coords, colours, filename):
    """
    将点云数据保存为 PLY 文件。
    :param coords: (N, 3) 点云坐标
    :param colours: (N, 3) RGB 颜色
    :param filename: 要保存的 PLY 文件名
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"保存成功: {filename}")

import torch
import torch.nn as nn
import MinkowskiEngine as ME


class MyDeepNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MyDeepNet, self).__init__()

        # 第一层卷积块
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, 32, kernel_size=3, dimension=3
        )
        self.norm1 = ME.MinkowskiBatchNorm(32)
        self.relu1 = ME.MinkowskiSigmoid()

        # 第一组卷积块
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiSigmoid(),
        )

        # 第二组卷积块
        self.block2 = nn.Sequential(
            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiSigmoid(),
        )

        # 第三组卷积块
        self.block3 = nn.Sequential(
            ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiSigmoid(),
        )

        # 第四组卷积块
        self.block4 = nn.Sequential(
            ME.MinkowskiConvolution(256, 512, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiSigmoid(),
        )

        # 加入第五组卷积块（加深网络结构）
        self.block5 = nn.Sequential(
            ME.MinkowskiConvolution(512, 1024, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(1024),
            ME.MinkowskiSigmoid(),
        )

        # 第六组卷积块（进一步加深网络结构）
        self.block6 = nn.Sequential(
            ME.MinkowskiConvolution(1024, 2048, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(2048),
            ME.MinkowskiSigmoid(),
        )

        # # 第七组卷积块（进一步加深网络结构）
        # self.block7 = nn.Sequential(
        #     ME.MinkowskiConvolution(2048, 4096, kernel_size=3, stride=1, dimension=3),
        #     ME.MinkowskiBatchNorm(4096),
        #     ME.MinkowskiSigmoid(),
        # )

        # 最后的卷积层输出
        self.conv_out = ME.MinkowskiConvolution(2048, out_channels, kernel_size=1, dimension=3)
        self.relu = ME.MinkowskiReLU()

    def forward(self, x):
        # 第一层卷积
        x = self.relu(self.norm1(self.conv1(x)))

        # 第一组卷积块
        x = self.block1(x)

        # 第二组卷积块
        x = self.block2(x)

        # 第三组卷积块
        x = self.block3(x)

        # 第四组卷积块
        x = self.block4(x)

        # 第五组卷积块
        x = self.block5(x)

        # 第六组卷积块
        x = self.block6(x)

        # # 第七组卷积块
        # x = self.block7(x)

        # 最后的卷积输出
        x = self.conv_out(x)
        return x

def position_loss(pred, target):
    if isinstance(pred, ME.SparseTensor) and isinstance(target, ME.SparseTensor):
        # 使用稀疏张量的密集特征进行损失计算
        return torch.nn.functional.mse_loss(pred.F, target.F)
    else:
        # 假设 pred 和 target 都是普通张量
        return torch.nn.functional.mse_loss(pred, target)

def train_model(model, data_loader, optimizer,device='cuda', epochs=10, blocks_per_epoch=8, lr=0.001):
    log_file = 'My-net_residual_training_log.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    model = model.to(device).float()
    model.train()

    # 用于记录每个epoch的平均损失
    epoch_losses = []

    for epoch in range(epochs):
        print(f'开始第{epoch}轮次')
        total_loss = 0
        num_batches = 0
        block_buffer_A, block_buffer_B = [], []

        for (chunks_A, chunks_B) in data_loader:
            block_buffer_A.extend(chunks_A)
            block_buffer_B.extend(chunks_B)

            if len(block_buffer_A) >= blocks_per_epoch:
                # 每blocks_per_epoch个块合并成一个批次
                for i in range(0, len(block_buffer_A), blocks_per_epoch):
                    # 提取坐标并合并
                    coords_A_batch = []
                    coords_B_batch = []

                    # 遍历当前批次的块，最多blocks_per_epoch个
                    for j in range(min(blocks_per_epoch, len(block_buffer_A) - i)):
                        coords_A_batch.append(block_buffer_A[i + j][:, :3])
                        coords_B_batch.append(block_buffer_B[i + j][:, :3])

                    # 确保 coords_A_batch 和 coords_B_batch 是 float 类型
                    coords_A_batch = [coord.float() for coord in coords_A_batch]
                    coords_B_batch = [coord.float() for coord in coords_B_batch]

                    coords_A_batch = torch.cat(coords_A_batch, dim=0)
                    coords_B_batch = torch.cat(coords_B_batch, dim=0)

                    # 合并坐标，确保数据类型为 float64
                    coords_A_tensor = ME_utils.batched_coordinates([coords_A_batch.view(-1, 3)], device=device)
                    coords_B_tensor = ME_utils.batched_coordinates([coords_B_batch.view(-1, 3)], device=device)

                    # 使用坐标作为特征
                    features_A = normalize_coordinates(coords_A_batch.numpy(), global_min, global_max)
                    features_B = normalize_coordinates(coords_B_batch.numpy(), global_min, global_max)

                    # Convert back to tensor
                    features_A = torch.tensor(features_A, dtype=torch.float32).to(device)
                    features_B = torch.tensor(features_B, dtype=torch.float32).to(device)

                    # 创建一个坐标管理器
                    coordinate_manager = ME.CoordinateManager(D=3)

                    origin = ME.SparseTensor(features=features_A, coordinates=coords_A_tensor, coordinate_manager=coordinate_manager)

                    # 构造 SparseTensor
                    input_x = ME.SparseTensor(features=features_B, coordinates=coords_B_tensor, coordinate_manager=coordinate_manager)

                    # 确保 new_x 是 float32 类型
                    input_x = ME.SparseTensor(
                        features=input_x.F.float(),
                        coordinates=input_x.C,
                        coordinate_manager=input_x.coordinate_manager
                    )

                    # 如果形状不一致，裁剪形状
                    if input_x.shape[0] != origin.shape[0]:
                        min_shape = min(input_x.shape[0], origin.shape[0])
                        inputs = ME.SparseTensor(
                            features=input_x.F[:min_shape].float(),
                            coordinates=input_x.C[:min_shape],
                            coordinate_manager=input_x.coordinate_manager

                        )
                        origin = ME.SparseTensor(
                            features=origin.F[:min_shape].float(),
                            coordinates=origin.C[:min_shape],
                            coordinate_manager=origin.coordinate_manager

                        )
                    else:
                        inputs = input_x

                    # 计算损失
                    output = model(inputs)
                    # 计算残差
                    residual = origin.F - inputs.F
                    print(output.shape,origin.shape,input_x.shape,residual.shape)

                    # 计算损失
                    loss = position_loss(output.F.float(), residual.float())
                    print(output.F.float(), residual.float())
                    # 累加损失
                    total_loss += loss.item()
                    num_batches += 1

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 清空缓冲区
                block_buffer_A.clear()
                block_buffer_B.clear()

        # 每个epoch结束时计算并记录平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)
        print(epoch_losses)

        # 每第10个epoch保存模型权重
        if (epoch + 1) % 10 == 0:
            save_path = './model/my_net_residual/' + str(epoch + 1) + '_model_residual.pth'
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at epoch {epoch + 1} to {save_path}")

    # 训练完成后，绘制损失-epoch图
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), epoch_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()

    # 保存损失图
    plt.savefig('My_Net_residual_Loss_vs_Epoch.png')
    plt.show()


def main(mode):
    if mode == 'test_process':
        folder_A = './data_sample/soldier/original'
        folder_B = './data_sample/soldier/compress'
        output_folder = './data_sample/soldier/block'
        process = True
    elif mode == 'full_process':
        folder_A = './data30/soldier/original'
        folder_B = './data30/soldier/compress'
        output_folder = './data30/soldier/block'
        process = True
    elif mode == 'test_run':
        folder_A = './data_sample/soldier/block/adjusted_chunks_A'
        folder_B = './data_sample/soldier/block/adjusted_chunks_B'
        output_folder = ''
        process = False
    elif mode == 'full_run':
        folder_A = './data30/soldier/block/adjusted_chunks_A'
        folder_B = './data30/soldier/block/adjusted_chunks_B'
        output_folder = ''
        process = False

    else:
        raise ValueError("Invalid mode. Use 'test' or 'full'.")
    dataset = PointCloudDataset(folder_A=folder_A,
                                folder_B=folder_B,
                                output_folder=output_folder,
                                block_size = 64,
                                cube_size = 1024,
                                min_points_ratio=0.00001,
                                mode=mode,
                                preprocess=process
                                )

    data_loader = DataLoader(dataset, batch_size=1)
    model = MyDeepNet()
    # model = MyTestNet()
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)  # 使用较小标准差的正态分布初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(weights_init)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # 训练模型
    train_model(model, data_loader, optimizer,epochs=100,blocks_per_epoch=8)


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Run training or testing mode.")
        parser.add_argument("mode", choices=["test_process", "full_process", "test_run", "full_run"],
                            help="Mode to run: 'test' or 'full'")
        args = parser.parse_args()

        main(args.mode)
