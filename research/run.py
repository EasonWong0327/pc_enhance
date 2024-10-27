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
import pandas as pd
import logging


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
    coords, colors = points[:, :3], points[:, 3:]

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
                block_colors = colors[mask]
                if len(block_coords) >= min_points_threshold:
                    block_points = torch.cat((block_coords, block_colors), dim=1)
                    blocks.append((block_points.cpu().numpy(), (x.item(), y.item(), z.item())))

                # 清理未使用的张量
                del block_coords, block_colors
                torch.cuda.empty_cache()
    print(f"总切块数: {len(blocks)}")  # 添加打印信息
    return blocks


# def adjust_points(chunk_A, chunk_B, n_points):
#     """
#     调整chunk_A中的点以匹配chunk_B中的点，并确保没有重复点。
#
#     参数：
#         chunk_A (np.ndarray): chunk A中的点数组 (N x 3或N x 6，具体取决于维度)。
#         chunk_B (np.ndarray): chunk B中的点数组 (M x 3或M x 6)。
#         n_points (int): 从chunk_A中选择的点数，以匹配chunk_B。
#
#     返回：
#         np.ndarray: 调整后的chunk_A，包含与chunk_B最接近的n_points个唯一点。
#     """
#     if len(chunk_A) == 0:
#         return np.empty((0, chunk_B.shape[1]))  # 返回空数组，保持维度一致
#
#     tree = KDTree(chunk_A)
#     adjusted_indices = set()  # 用于跟踪在chunk_A中唯一索引的集合
#
#     # 遍历chunk_B中的点，寻找在chunk_A中最近的唯一点
#     for point in chunk_B:
#         distances, indices = tree.query(point, k=len(chunk_A))  # 获取chunk_A中所有点的排序距离
#
#         # 寻找最近的未使用点
#         for index in indices:
#             if index not in adjusted_indices:
#                 adjusted_indices.add(index)
#                 break
#
#         # 如果已选择足够的点，则停止
#         if len(adjusted_indices) >= n_points:
#             break
#
#     # 将索引转换为列表并选择相应的点
#     adjusted_chunk_A = chunk_A[list(adjusted_indices)]
#
#     # 确保结果恰好有n_points个点（如有必要进行填充）
#     if len(adjusted_chunk_A) < n_points:
#         remaining_indices = list(set(range(len(chunk_A))) - adjusted_indices)
#         if remaining_indices:  # 检查是否还有剩余的索引可用
#             additional_indices = np.random.choice(
#                 remaining_indices,
#                 n_points - len(adjusted_chunk_A),
#                 replace=False
#             )
#             adjusted_chunk_A = np.vstack([adjusted_chunk_A, chunk_A[additional_indices]])
#
#     return adjusted_chunk_A


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
    def __init__(self, folder_A, folder_B, block_size=256, cube_size=1024,min_points_ratio=0.1):
        """
        初始化数据集，并确保文件编号匹配。
        :param folder_A: 原始点云A的文件夹路径
        :param folder_B: 压缩后点云B的文件夹路径
        :param chunk_size: 每个块的大小
        :param stride: 切块的步长
        """
        self.files_A = self.match_files(folder_A, folder_B)
        self.block_size = block_size
        self.cube_size = cube_size
        self.min_points_ratio = min_points_ratio

        if not self.files_A:
            print("没有找到匹配的文件对，请检查文件名和路径是否正确！")
        else:
            print(f"共找到 {len(self.files_A)} 对文件。")

    def match_files(self, folder_A, folder_B):
        """根据编号匹配压缩前后的文件对。"""

        # 获取两个文件夹中的所有 .ply 文件
        files_A = sorted([f for f in os.listdir(folder_A) if f.endswith('.ply')])
        files_B = sorted([f for f in os.listdir(folder_B) if f.endswith('.ply')])

        # 正则表达式：匹配文件名末尾的 3-4 位数字编号
        def extract_id(filename):
            match = re.search(r'(\d{3,4})(?=\.ply$)', filename)
            return match.group(1) if match else None

        # 创建以编号为键的文件路径映射
        files_A_dict = {extract_id(f): os.path.join(folder_A, f) for f in files_A}
        files_B_dict = {extract_id(f): os.path.join(folder_B, f) for f in files_B}

        # 打印匹配的文件编号字典，供调试用
        print("files_A_dict:", files_A_dict)
        print("files_B_dict:", files_B_dict)

        # 匹配两组文件编号的交集
        matched_files = [
            (files_A_dict[id_], files_B_dict[id_])
            for id_ in files_A_dict.keys() & files_B_dict.keys()
        ]

        if not matched_files:
            print("没有找到匹配的文件对，请检查文件名编号是否一致。")
        return matched_files

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, idx):
        file_A, file_B = self.files_A[idx]
        points_A = self.load_ply(file_A)
        points_B = self.load_ply(file_B)
        # 压缩后的ply有坐标重复
        check_compress = has_duplicates(points_B)
        if check_compress:
            points_B = remove_duplicates(points_B)
            # debug用
            # has_dup, duplicates = has_duplicates_output(points_B)
            # print(duplicates)
        chunks_A = chunk_point_cloud_fixed_size(points_A, self.block_size, self.cube_size, self.min_points_ratio)
        chunks_B = chunk_point_cloud_fixed_size(points_B, self.block_size, self.cube_size, self.min_points_ratio)
        print(f"文件对 {idx}： A 切块数: {len(chunks_A)}，B 切块数: {len(chunks_B)}")  # 添加打印信息
        adjusted_chunks_A = []
        adjusted_chunks_B = []

        for (chunk_A, index_A), (chunk_B, index_B) in zip(chunks_A, chunks_B):
            if index_A == index_B:
                print('未补齐之前chunk_B是否有重复：',has_duplicates(chunk_B))
                # 根据chunk的大小调整A或B
                if len(chunk_A) < len(chunk_B):
                    # adjusted_chunk_B = adjust_points(chunk_B, chunk_A, n_points=len(chunk_A)) # 调整B以匹配A
                    adjusted_chunk_B = adjust_points(chunk_B, chunk_A) # 调整B以匹配A
                    print(
                        f"调整前的 chunk_B 点数: {chunk_B.shape[0]}，调整后的点数: {adjusted_chunk_B.shape[0]}，chunk_A的点数: {chunk_A.shape[0]}")

                    adjusted_chunks_B.append(adjusted_chunk_B)
                    adjusted_chunks_A.append(chunk_A)
                else:
                    start_time = time.time()
                    # adjusted_chunk_A = adjust_points(chunk_A, chunk_B, n_points=len(chunk_B))
                    adjusted_chunk_A = adjust_points(chunk_A, chunk_B)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"KD-Tree耗时: {elapsed_time:.4f} 秒")
                    print(f"调整前的 chunk_A 点数: {chunk_A.shape[0]}，调整后的点数: {adjusted_chunk_A.shape[0]}，chunk_B的点数: {chunk_B.shape[0]}")
                    adjusted_chunks_A.append(adjusted_chunk_A)
                    adjusted_chunks_B.append(chunk_B)
        if not adjusted_chunks_A or not adjusted_chunks_B:
            print(f"第 {idx} 对文件没有找到匹配的块，A: {len(adjusted_chunks_A)} 块, B: {len(adjusted_chunks_B)} 块.")
            return None
        print(f"第 {idx} 对文件切块完成，A: {len(adjusted_chunks_A)} 块, B: {len(adjusted_chunks_B)} 块. ",
              "其中一块的形状:", adjusted_chunks_A[0].shape)
        print('-------------------------------开始打印每块的点数------------------------')
        for i in range(len(adjusted_chunks_A)):
            print(f'第{i}块: ', adjusted_chunks_A[i].shape)

        return (adjusted_chunks_A, adjusted_chunks_B)

    def load_ply(self, file_path):
        """读取 PLY 文件并返回 (N, 6) 的点云数组"""
        print(f"正在加载文件：{file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(f"加载点数量：{points.shape[0]}")
        return np.hstack((points, colors))


def save_point_cloud_as_ply(coords, feats, filename):
    """
    将点云数据保存为 PLY 文件。
    :param coords: (N, 3) 点云坐标
    :param feats: (N, 3) RGB 颜色
    :param filename: 要保存的 PLY 文件名
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(feats / 255.0)  # 归一化颜色

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"保存成功: {filename}")



class MyTestNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels=3, out_channels=3, D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm1 = ME.MinkowskiBatchNorm(out_channels)
        self.relu1 = ME.MinkowskiReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        return out


if __name__ == '__main__':
    dataset = PointCloudDataset(folder_A='./data/original',
                                folder_B='./data/compress',
                                block_size = 128,
                                cube_size = 1024,
                                min_points_ratio=0.0001
                                )

    data_loader = DataLoader(dataset, batch_size=1)


    # -----------------------------run train test -------------------------------
    # -----------------------------要去服务器执行，本地Windows没装ME -------------------------------
    def position_loss(pred, target):
        if isinstance(pred, ME.SparseTensor) and isinstance(target, ME.SparseTensor):
            # 使用稀疏张量的密集特征进行损失计算
            return torch.nn.functional.mse_loss(pred.F, target.F)
        else:
            # 假设 pred 和 target 都是普通张量
            return torch.nn.functional.mse_loss(pred, target)


    def train_model(model, data_loader, optimizer, device='cuda', epochs=10, blocks_per_epoch=1):

        log_file = 'training_log.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()

        model = model.to(device)
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            block_buffer_A, block_buffer_B = [], []

            for (chunks_A, chunks_B) in data_loader:
                block_buffer_A.extend(chunks_A)
                block_buffer_B.extend(chunks_B)

                if len(block_buffer_A) >= blocks_per_epoch:
                    # 每N个块进行处理
                    for i in range(0, len(block_buffer_A), blocks_per_epoch):
                        if i + 1 >= len(block_buffer_A) or i + 1 >= len(block_buffer_B):
                            # print(f"Skipping block {i + 1} as it exceeds buffer length.")
                            break

                        # print(block_buffer_A[i].shape, block_buffer_B[i].shape)
                        logger.info(f"Processing block {i} and {i + 1}")

                        # 提取坐标和特征
                        coords_A1, feats_A1 = block_buffer_A[i][0, :, :3], block_buffer_A[i][0, :, 3:]
                        coords_A2, feats_A2 = block_buffer_A[i + 1][0, :, :3], block_buffer_A[i + 1][0, :, 3:]
                        coords_B1, feats_B1 = block_buffer_B[i][0, :, :3], block_buffer_B[i][0, :, 3:]
                        coords_B2, feats_B2 = block_buffer_B[i + 1][0, :, :3], block_buffer_B[i + 1][0, :, 3:]

                        # print('coords_A1_shape:', coords_A1.shape, coords_A2.shape, coords_B1.shape, coords_B2.shape)

                        # 合并坐标和特征
                        coords_A_tensor = ME_utils.batched_coordinates([coords_A1.view(-1, 3), coords_A2.view(-1, 3)],
                                                                       device=device)
                        feats_A_tensor = torch.cat([feats_A1, feats_A2]).to(device).float()

                        coords_B_tensor = ME_utils.batched_coordinates([coords_B1.view(-1, 3), coords_B2.view(-1, 3)],
                                                                       device=device)
                        feats_B_tensor = torch.cat([feats_B1, feats_B2]).to(device).float()

                        # print('coords_AB_tensor_shape:', coords_A_tensor.shape, coords_B_tensor.shape)
                        # print('feats_AB_tensor_shape:', feats_A_tensor.shape, feats_B_tensor.shape)

                        # 确保行数一致
                        assert feats_B_tensor.shape[0] == coords_B_tensor.shape[
                            0], "Feature and coordinates row count must match!"
                        assert feats_A_tensor.shape[0] == coords_A_tensor.shape[
                            0], "Feature and coordinates row count must match!"

                        origin = ME.SparseTensor(features=feats_A_tensor, coordinates=coords_A_tensor)
                        # print('model_origin_shape:', origin.shape)

                        # 构造 SparseTensor
                        x = ME.SparseTensor(features=feats_B_tensor, coordinates=coords_B_tensor)
                        # print('model_input_x_shape:', x.shape, type(x))

                        # 确保 new_x 是 float32 类型
                        x = ME.SparseTensor(
                            features=x.F.float(),
                            coordinates=x.C,
                            coordinate_manager=x.coordinate_manager
                        )

                        optimizer.zero_grad()
                        output = model(x)

                        # # 计算残差并合并输出
                        # residual = output.F.float() - feats_B_tensor.float()
                        # combined_output = output.F + residual

                        # 计算损失 模型output与origin进行计算
                        loss = position_loss(output.F.float(), origin.F.float())
                        total_loss += loss.item()

                        loss.backward()
                        optimizer.step()
                        batch_loss_log = f"Epoch {epoch + 1}/{epochs}, Batch {num_batches + 1}, Loss: {loss.item():.4f}"
                        logger.info(batch_loss_log)

                    avg_loss = total_loss / (blocks_per_epoch // 2)
                    avg_loss_log = f"Epoch {epoch + 1}/{epochs}, Batch {num_batches + 1}, Average Loss: {avg_loss:.4f}"
                    logger.info(avg_loss_log)

                    # 清空缓冲区
                    block_buffer_A.clear()
                    block_buffer_B.clear()
                    total_loss = 0
                    num_batches += 1
            # 保存模型权重
            save_path = str(epoch) + '_model.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")
            model_save_log = f"Model saved at epoch {epoch + 1} to {save_path}"
            logger.info(model_save_log)

    # model = MyNet()

    model = MyTestNet()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, data_loader, optimizer,epochs=1,blocks_per_epoch=8)
