import numpy as np
from scipy.spatial import KDTree

def has_duplicates(points, tol=1e-9):
    """
    检查点云数据集中是否存在重复的点。

    参数:
    points (numpy.ndarray): 点云数据，形状为 (N, D)。
    tol (float): 判断重复点的容差，默认为 1e-9。

    返回:
    bool: 如果存在重复的点，则返回 True；否则返回 False。
    """
    tree = KDTree(points)
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=2)
        if distances[1] < tol:
            return True
    return False

points = np.random.rand(100, 3)
print("是否有重复点:", has_duplicates(points))