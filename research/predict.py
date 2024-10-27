import os
import torch
import re
import open3d as o3d  # 用于读取 PLY 文件
import numpy as np
import MinkowskiEngine as ME

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


def chunk_point_cloud_fixed_size(points, block_size=256, cube_size=1024, min_points_ratio=0.1, device='cuda'):
    """
    单独摘出来，本地调试没有ME
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


def match_files(folder_A, folder_B):
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

def load_ply(file_path):
    """读取 PLY 文件并返回 (N, 6) 的点云数组"""
    print(f"正在加载文件：{file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"加载点数量：{points.shape[0]}")
    return np.hstack((points, colors))


def save_blocks_as_ply_with_open3d(blocks, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(file_name)[0]

    for i, (block_points, (x, y, z)) in enumerate(blocks):
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(block_points[:, :3])

        if block_points.shape[1] >= 6:
            colors = (block_points[:, 3:6] / 255.0).astype(np.float32)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            print(f"块 {i} 没有色彩信息，将只保存坐标。")

        # 创建合法的文件名
        filename = os.path.join(save_dir, f"{base_name}_block_{i}_at_{x}_{y}_{z}.ply")

        try:
            o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
            print(f"保存块 {i} 为文件 {filename}")
        except Exception as e:
            print(f"保存失败: {e}")  # 捕获异常并输出错误信息


def load_model_for_prediction(model, model_path='model.pth', device='cuda'):
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def predict_on_blocks(model, blocks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型放在GPU上
    with torch.no_grad():
        predictions = []
        for block_points, (x, y, z) in blocks:
            # 将块数据传输到GPU上，并确保坐标为整数类型
            coords_B = torch.tensor(block_points[:, :3], dtype=torch.int).to(device)
            feats_B = torch.tensor(block_points[:, 3:], dtype=torch.float32).to(device)
            coords_B = torch.cat([torch.zeros((coords_B.shape[0], 1), dtype=torch.int).to(device), coords_B], dim=1)
            inputs = ME.SparseTensor(features=feats_B, coordinates=coords_B)  # 移除 dimension 参数

            output = model(inputs)
            # # 将原始颜色与模型输出的坐标合并(单独任务，暂时用不到，而且有重复的点)
            # predicted_points = np.hstack((output.F.cpu().numpy(), block_points[:, 3:]))  # 预测的坐标和原始颜色
            # predictions.append((predicted_points, (x, y, z)))  # 保留预测输出和坐标
            print("Prediction output shape:", output.F.shape)
            predictions.append((output.F.cpu().numpy(), (x, y, z)))  # 保留预测输出和坐标
    return predictions


def main(origin_dir, compress_dir, model_path, output_dir,
         block_size,cube_size,min_points_ratio):
    # 加载训练好的模型并进行预测
    model = MyTestNet()
    model = model.float()
    # 加载模型权重
    model = load_model_for_prediction(model, model_path, device='cuda')

    # 创建输出文件夹
    origin_block_dir = os.path.join(output_dir, 'origin_block')
    compress_block_dir = os.path.join(output_dir, 'compress_block')
    predict_block_dir = os.path.join(output_dir, 'model_predict_block')
    os.makedirs(origin_block_dir, exist_ok=True)
    os.makedirs(compress_block_dir, exist_ok=True)
    os.makedirs(predict_block_dir, exist_ok=True)

    # 匹配文件对
    files_pair = match_files(origin_dir, compress_dir)
    print(files_pair)

    for file_A, file_B in files_pair:
        # 提取文件名部分，用于输出文件的命名
        base_name_A = os.path.basename(file_A)
        base_name_B = os.path.basename(file_B)

        # 加载并切分文件 A 和 B
        points_A = load_ply(file_A)
        points_B = load_ply(file_B)
        chunks_A = chunk_point_cloud_fixed_size(points_A, block_size, cube_size, min_points_ratio)
        chunks_B = chunk_point_cloud_fixed_size(points_B, block_size, cube_size, min_points_ratio)

        # 保存切分的块
        save_blocks_as_ply_with_open3d(chunks_A, origin_block_dir, base_name_A)
        save_blocks_as_ply_with_open3d(chunks_B, compress_block_dir, base_name_B)

        predicted_blocks = predict_on_blocks(model, chunks_B)
        save_blocks_as_ply_with_open3d(predicted_blocks, predict_block_dir, base_name_B)

if __name__ == "__main__":
    origin_dir = 'test/original'
    compress_dir = 'test/compress'
    model_path = '0_model.pth'
    output_dir = 'test/output'
    block_size = 8  # 根据实际情况调整block大小
    main(origin_dir, compress_dir, model_path, output_dir,
         block_size=128,cube_size=1024,min_points_ratio=0.0001)


