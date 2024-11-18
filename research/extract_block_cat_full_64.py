# 把kd-tree拆出来
from collections import defaultdict
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import argparse
import shutil
import time

def remove_duplicates(point_cloud):
    """
    去重点云坐标

    参数:
    point_cloud (numpy.ndarray): 点云坐标，形状为 (N, 3)

    返回:
    numpy.ndarray: 去重后的点云坐标
    """
    # 使用np.unique去重，并保持原有顺序
    unique_points = np.unique(point_cloud, axis=0)
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


def chunk_point_cloud_fixed_size(points, block_size=256, cube_size=1024, overlap=1, device='cuda'):
    """
    将点云数据切分为固定大小的块，支持可选偏移，并确保块的数量和大小一致。
    """
    points = torch.tensor(points, device=device, dtype=torch.float32)  # 使用 float32
    coords = points

    stride = block_size - overlap  # 步长，考虑重叠区域
    x_range = torch.arange(0, cube_size, stride, device=device)
    y_range = torch.arange(0, cube_size, stride, device=device)
    z_range = torch.arange(0, cube_size, stride, device=device)

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
                if len(block_coords) >= 0:
                    blocks.append((block_coords.cpu().numpy(), (x.item(), y.item(), z.item())))

                # 清理未使用的张量
                del block_coords
                torch.cuda.empty_cache()
    
    print(f"总切块数: {len(blocks)}")  # 添加打印信息
    return blocks


# def adjust_points(chunk_A, chunk_B):
#     if len(chunk_A) == 0 or len(chunk_B) == 0:
#         return np.empty((0, chunk_B.shape[1]))  # 返回空数组，保持维度一致

#     n_points = len(chunk_B)  # 设置n_points为chunk_B的长度
#     tree = KDTree(chunk_A)
#     adjusted_indices = set()  # 用于跟踪在chunk_A中唯一索引的集合
#     adjusted_chunk_A = np.empty((n_points, chunk_A.shape[1]))  # 初始化调整后的chunk_A

#     # 遍历chunk_B中的点，寻找在chunk_A中最近的唯一点
#     for i, point in enumerate(chunk_B):
#         distances, indices = tree.query(point, k=len(chunk_A))  # 获取chunk_A中所有点的排序距离

#         # 寻找最近的未使用点
#         for index in indices:
#             if index not in adjusted_indices:
#                 adjusted_indices.add(index)
#                 adjusted_chunk_A[i] = chunk_A[index]
#                 break

#     return adjusted_chunk_A


def adjust_points(chunk_A, chunk_B):
    n_points = len(chunk_B)  # 设置n_points为chunk_B的长度

    # 创建Open3D点云对象
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(chunk_A)

    # 构建KDTree
    tree = o3d.geometry.KDTreeFlann(pcd_A)
    adjusted_indices = set()  # 用于跟踪在chunk_A中唯一索引的集合
    adjusted_chunk_A = np.empty((n_points, chunk_A.shape[1]))  # 初始化调整后的chunk_A

    # 遍历chunk_B中的点，寻找在chunk_A中最近的唯一点
    for i, point in enumerate(chunk_B):
        [_, indices, _] = tree.search_knn_vector_3d(point, len(chunk_A))  # 获取chunk_A中所有点的排序距离

        # 寻找最近的未使用点
        for index in indices:
            if index not in adjusted_indices:
                adjusted_indices.add(index)
                adjusted_chunk_A[i] = chunk_A[index]
                break

    return adjusted_chunk_A

class PointCloudDataset(Dataset):
    def __init__(self, folder_A, folder_B, block_folder,merged_folder, block_size, cube_size, min_points_ratio,mode=''):
        # path
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.block_folder = block_folder
        self.merged_folder = merged_folder

        # size
        self.block_size = block_size
        self.cube_size = cube_size
        self.min_points_ratio = min_points_ratio

        # status
        self.mode = mode

        self.file_pairs = self._get_file_pairs()
        self._preprocess_data()
        self._cat_block()

    def _cat_block(self):
        folder_A = f'{self.block_folder}/adjusted_chunks_A'
        mergerd_folder_A = str(folder_A).replace('adjusted_chunks_A','merged_original')
        if os.path.exists(mergerd_folder_A):
            shutil.rmtree(mergerd_folder_A)
        os.makedirs(mergerd_folder_A, exist_ok=True)
        self.merge_blocks(folder_A, mergerd_folder_A)

        folder_B = f'{self.block_folder}/adjusted_chunks_B'
        mergerd_folder_B = str(folder_B).replace('adjusted_chunks_B','merged_compress')
        if os.path.exists(mergerd_folder_B):
            shutil.rmtree(mergerd_folder_B)
        os.makedirs(mergerd_folder_B, exist_ok=True)
        self.merge_blocks(folder_B, mergerd_folder_B)


    def merge_blocks(self, block_folder, output_folder):
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
                block_points = self._load_ply(block_file)
                all_points.append(block_points)

            # 合并所有块
            all_points = np.vstack(all_points)

            # 确定输出文件名
            if 'vox10' in key:
                output_file = os.path.join(output_folder, f"{key}_origin.ply")
            elif 'S26C03R03_rec' in key:
                output_file = os.path.join(output_folder, f"{key}.ply")

            # 保存为一个完整的PLY文件
            self._save_ply(all_points, output_file)
            print(f"合并后的点云保存为: {output_file}")

    def _get_file_pairs(self):
        # 获取folder_A和folder_B中的文件对
        files_A = sorted([f for f in os.listdir(self.folder_A) if f.endswith('.ply')])
        files_B = sorted([f for f in os.listdir(self.folder_B) if f.endswith('.ply')])
        return list(zip(files_A, files_B))

    def _preprocess_data(self):
        # 创建保存adjusted_chunks_A和adjusted_chunks_B的文件夹

        adjusted_A_folder = os.path.join(self.block_folder, 'adjusted_chunks_A')
        adjusted_B_folder = os.path.join(self.block_folder, 'adjusted_chunks_B')
        if os.path.exists(adjusted_A_folder):
            shutil.rmtree(adjusted_A_folder)
        if os.path.exists(adjusted_B_folder):
            shutil.rmtree(adjusted_B_folder)
        os.makedirs(adjusted_A_folder, exist_ok=True)
        os.makedirs(adjusted_B_folder, exist_ok=True)

        # 遍历文件对并进行kd-tree匹配
        for file_A, file_B in self.file_pairs:
            # 加载并处理点云文件
            print('开始处理：', self.folder_A, file_A, file_B)
            points_A = self._load_ply(os.path.join(self.folder_A, file_A))
            points_B = self._load_ply(os.path.join(self.folder_B, file_B))
            print(f"压缩点云点数: {points_B.shape[0]}")
            # 1.数据清洗：去除重复坐标   压缩后的ply有坐标重复
            points_B = remove_duplicates(points_B)
            print(f"去重后的点云点数: {points_B.shape[0]}")
            # debug用
            # has_dup, duplicates = has_duplicates_output(points_B)
            # print(duplicates)
            # 2.数据切块
            chunks_A = chunk_point_cloud_fixed_size(points_A, self.block_size, self.cube_size)
            chunks_B = chunk_point_cloud_fixed_size(points_B, self.block_size, self.cube_size)

            if self.mode == 'test_process':
                chunks_A = chunks_A[0:8]
                chunks_B = chunks_B[0:8]

            # 3.kd-tree匹配
            adjusted_chunks_A = []
            adjusted_chunks_B = []
            for (chunk_A, index_A), (chunk_B, index_B) in zip(chunks_A, chunks_B):
                if index_A == index_B and len(chunk_B)>0 and len(chunk_A)>0:
                    # 按道理，应该chunk_A>chunk_B
                    if len(chunk_A) < len(chunk_B):
                        adjusted_chunk_B = adjust_points(chunk_B, chunk_A)  # 调整B以匹配A
                        print(
                            f"调整前的 chunk_B 点数: {chunk_B.shape[0]}，调整后的点数: {adjusted_chunk_B.shape[0]}，chunk_A的点数: {chunk_A.shape[0]}")
                        adjusted_chunks_B.append(adjusted_chunk_B)
                        adjusted_chunks_A.append(chunk_A)
                    else:
                        adjusted_chunk_A = adjust_points(chunk_A, chunk_B)
                        print(
                            f"调整前的 chunk_A 点数: {chunk_A.shape[0]}，调整后的点数: {adjusted_chunk_A.shape[0]}，chunk_B的点数: {chunk_B.shape[0]}")
                        adjusted_chunks_A.append(adjusted_chunk_A)
                        adjusted_chunks_B.append(chunk_B)

            print('-------------------------------开始打印每块的点数------------------------')
            for i in range(len(adjusted_chunks_A)):
                print(f'第{i}块: ', adjusted_chunks_A[i].shape, adjusted_chunks_B[i].shape)

                file_A = file_A.replace('.ply', '')
                file_B = file_B.replace('.ply', '')
                self._save_ply(adjusted_chunks_A[i], os.path.join(adjusted_A_folder, f"{file_A}_block_{i}.ply"))
                self._save_ply(adjusted_chunks_B[i], os.path.join(adjusted_B_folder, f"{file_B}_block_{i}.ply"))

    def _load_ply(self, file_path):
        # 只要坐标
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        return points

    def _save_ply(self, points, file_path):
        # 使用open3d保存点云数据为ply格式，不包含颜色
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)  # 只保存坐标信息
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


def main(mode):
    process = True
    base_folder = './data30/soldier'

    folder_A = f'{base_folder}/original'
    folder_B = f'{base_folder}/compress'
    block_folder = f'{base_folder}/block64'
    merged_folder = f'{base_folder}/merged64'
    dataset = PointCloudDataset(folder_A=folder_A,
                                folder_B=folder_B,
                                block_folder=block_folder,
                                merged_folder=merged_folder,
                                block_size=64,
                                cube_size=1024,
                                min_points_ratio=0,
                                mode=mode,
                                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training or testing mode.")
    parser.add_argument("mode", choices=["test_process", "full_process", "test_run", "full_run"],
                        help="Mode to run: 'test' or 'full'")
    args = parser.parse_args()

    main(args.mode)