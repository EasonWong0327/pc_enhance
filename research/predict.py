import os
import torch
import re
import open3d as o3d  # 用于读取 PLY 文件
import numpy as np
import MinkowskiEngine as ME
import subprocess

device = torch.device('cuda:0')


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


def chunk_point_cloud_fixed_size(points, block_size=256, cube_size=1024, device=device):
    """
    将点云数据切分为固定大小的块，支持可选偏移，并确保块的数量和大小一致。
    """
    points = torch.tensor(points, device=device, dtype=torch.float32)  # 使用 float32
    coords, colors = points[:, :3], points[:, 3:]

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

                if block_coords.shape[0] == 0:
                    continue  # 跳过空块

                # 去除重复点，仅保留唯一坐标
                unique_coords = torch.unique(block_coords, dim=0)

                # 将去重后的坐标保存到块列表
                blocks.append((unique_coords.cpu().numpy(), (x.item(), y.item(), z.item())))

                # 清理未使用的张量
                del block_coords, unique_coords
                torch.cuda.empty_cache()
    print(f"总切块数: {len(blocks)}")  # 添加打印信息
    return blocks


def load_ply(file_path):
    """读取 PLY 文件并返回 (N, 6) 的点云数组"""
    print(f"正在加载文件：{file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"加载点数量：{points.shape[0]}")
    return np.hstack((points, colors))


def save_ply_with_open3d(points, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] >= 6:
        colors = (points[:, 3:6] / 255.0).astype(np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)
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
    return denormalized_coords


def predict_on_blocks(model, blocks, global_min, global_max):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型放在GPU上

    global_min = torch.tensor(global_min, dtype=torch.float32, device=device)
    global_max = torch.tensor(global_max, dtype=torch.float32, device=device)

    predictions = []
    for block_idx, (block_points, (x, y, z)) in enumerate(blocks):
        coords_B = torch.tensor(block_points[:, :3], dtype=torch.float32).to(device)
        feats_B = torch.tensor(block_points[:, 3:], dtype=torch.float32).to(device)

        normalized_coords_B = normalize_coordinates(coords_B, global_min, global_max)

        # 添加第四维用于批次索引
        int_coords_B = torch.cat([coords_B.int(), torch.full((coords_B.shape[0], 1), block_idx, dtype=torch.int32, device=device)], dim=1)

        inputs = ME.SparseTensor(features=normalized_coords_B, coordinates=int_coords_B)
        output = model(inputs)
        print('shape check:',feats_B.shape, inputs.shape,output.shape)
        # if output.F.shape[0] != normalized_coords_B.shape[0]:
        #     print(f"Warning: Mismatch in tensor sizes for block {block_idx}. Skipping this block.")
        #     continue  # 跳过点数不一致的块，避免错误
        print(normalized_coords_B)
        print(output.F)
        predicted_coords = normalized_coords_B + output.F

        denormalize_coords = denormalize_coordinates(predicted_coords, global_min, global_max)
        # 使用 detach() 去除梯度信息
        predicted_points = torch.cat((denormalize_coords, feats_B), dim=1).detach().cpu().numpy()
        predictions.append(predicted_points)

        # 清理未使用的张量
        del coords_B, feats_B, normalized_coords_B, inputs, output, predicted_coords, denormalize_coords
        torch.cuda.empty_cache()

    return np.vstack(predictions)


def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")

    # 使用 nvidia-smi 获取更详细的显存使用情况
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))


def main(compress_dir, model_path, output_dir,
         block_size, cube_size, global_min, global_max):
    if device.type == 'cuda':
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        free_memory = reserved_memory - allocated_memory

        print(f"当前使用的GPU索引: {torch.cuda.current_device()}")
        print(f"当前使用的GPU名称: {gpu_properties.name}")
        print(f"显存总容量: {total_memory:.2f} GB")
        print(f"已保留显存: {reserved_memory:.2f} GB")
        print(f"已分配显存: {allocated_memory:.2f} GB")
        print(f"空闲显存: {free_memory:.2f} GB")

    try:
        model = load_model_for_prediction(MyTestNet, model_path, device)
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
        chunks_B = chunk_point_cloud_fixed_size(points_B, block_size, cube_size)

        # 预测并合并所有块
        predicted_points = predict_on_blocks(model, chunks_B, global_min, global_max)

        # 保存合并后的点云
        output_file_path = os.path.join(output_dir, f"{base_name_B}_predicted.ply")
        save_ply_with_open3d(predicted_points, output_file_path)


if __name__ == "__main__":
    compress_dir = 'test30/compress'
    block_size = 64  # 根据实际情况调整block大小
    global_min = np.array([0.0, 0.0, 0.0])
    global_max = np.array([620.0, 1024.0, 662.0])
    cube_size = 1024  # 根据实际情况调整cube大小

    # 获取所有以 "_model.pth" 结尾的模型文件
    model_files = [f for f in os.listdir() if f.endswith('_model.pth')]

    for model_file in model_files:
        model_path = os.path.join(os.getcwd(), model_file)
        output_dir = f"test30/{model_file}_output"
        os.makedirs(output_dir, exist_ok=True)

        # 调用 main 函数进行预测
        main(compress_dir, model_path, output_dir,
             block_size=block_size, cube_size=cube_size, global_min=global_min, global_max=global_max)

