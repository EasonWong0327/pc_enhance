import open3d as o3d
import numpy as np

def calculate_ply_dimensions(ply_file):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_file)

    # 提取点云数据
    points = np.asarray(pcd.points)

    # 计算边界框
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # 计算尺寸
    dimensions = max_bound - min_bound

    # 计算中心位置
    center = (max_bound + min_bound) / 2

    return min_bound, max_bound, dimensions, center

ply_file = "./data/original/soldier_vox10_0536.ply"
min_bound, max_bound, dimensions, center = calculate_ply_dimensions(ply_file)

print("最小边界:", min_bound)
print("最大边界:", max_bound)
print("尺寸:", dimensions)
print("中心位置:", center)
