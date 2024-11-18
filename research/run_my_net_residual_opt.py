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


# 自定义排序函数
def natural_sort_key(filename):
    # 使用正则表达式分割字符串，提取数字和非数字部分
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

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

    def _get_file_pairs(self):
        # 获取folder_A和folder_B中的文件对
        files_A = sorted([f for f in os.listdir(self.folder_A) if f.endswith('.ply')],key=natural_sort_key)
        files_B = sorted([f for f in os.listdir(self.folder_B) if f.endswith('.ply')],key=natural_sort_key)
        return list(zip(files_A, files_B))

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

def position_loss(pred, target):
    if isinstance(pred, ME.SparseTensor) and isinstance(target, ME.SparseTensor):
        # 使用稀疏张量的密集特征进行损失计算
        return torch.nn.functional.mse_loss(pred.F, target.F)
    else:
        # 假设 pred 和 target 都是普通张量
        return torch.nn.functional.mse_loss(pred, target)

def train_model(model, data_loader, optimizer,device='cuda', epochs=10, blocks_per_epoch=8, lr=0.001):
    log_file = 'My-net_residual_training_log1117.log'
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
#                     print(input_x.shape)
#                     print(input_x)

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
#                     print(output.shape,origin.shape,input_x.shape,residual.shape)

                    # 计算损失
                    loss = position_loss(output.F.float(), residual.float())
#                     print(output.F.float(), residual.float())
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
            save_path = './model/my_net_residual_1117/' + str(epoch + 1) + '_model_residual.pth'
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
    plt.savefig('My_Net_residual_Loss_1117.png')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    train_model(model, data_loader, optimizer,epochs=200,blocks_per_epoch=32)


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Run training or testing mode.")
        parser.add_argument("mode", choices=["test_process", "full_process", "test_run", "full_run"],
                            help="Mode to run: 'test' or 'full'")
        args = parser.parse_args()

        main(args.mode)