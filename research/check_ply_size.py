import os
import numpy as np
import open3d as o3d

def get_point_cloud_min_max(file_path):
    """
    读取PLY文件并计算点云数据的最小值和最大值。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    min_values = points.min(axis=0)
    max_values = points.max(axis=0)
    return min_values, max_values

def process_directory(directory):
    """
    遍历目录下的所有PLY文件，并计算它们的坐标最大值和最小值。
    """
    overall_min = np.array([np.inf, np.inf, np.inf])
    overall_max = np.array([-np.inf, -np.inf, -np.inf])

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ply'):
                file_path = os.path.join(root, file)
                min_values, max_values = get_point_cloud_min_max(file_path)
                overall_min = np.minimum(overall_min, min_values)
                overall_max = np.maximum(overall_max, max_values)

    return overall_min, overall_max

def process_multiple_directories(directories):
    """
    遍历多个目录下的所有PLY文件，并计算它们的坐标最大值和最小值。
    """
    overall_min = np.array([np.inf, np.inf, np.inf])
    overall_max = np.array([-np.inf, -np.inf, -np.inf])

    for directory in directories:
        dir_min, dir_max = process_directory(directory)
        overall_min = np.minimum(overall_min, dir_min)
        overall_max = np.maximum(overall_max, dir_max)

    return overall_min, overall_max

# 目录列表
directories = [
    '/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/soldier/Ply',
    '/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/longdress/Ply',
    '/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/loot/Ply',
    '/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/redandblack//Ply',

    '/home/jupyter-eason/data/software/mpeg-pcc-tmc2-master/output0911/redandblack_r1',
    '/home/jupyter-eason/data/software/mpeg-pcc-tmc2-master/output0911/soldier_r1'
]

overall_min, overall_max = process_multiple_directories(directories)


print("所有PLY文件的坐标最小值:")
print("x轴最小值:", overall_min[0])
print("y轴最小值:", overall_min[1])
print("z轴最小值:", overall_min[2])

print("\n所有PLY文件的坐标最大值:")
print("x轴最大值:", overall_max[0])
print("y轴最大值:", overall_max[1])
print("z轴最大值:", overall_max[2])