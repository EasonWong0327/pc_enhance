import os
import re
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from MinkowskiEngine import utils as ME_utils
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
from network import MyNet
import shutil
from collections import defaultdict
import subprocess
import pandas as pd


current_date = datetime.now()
formatted_date = str(current_date.strftime("%Y%m%d"))
task_name = 'austin_network_train_{}'.format(formatted_date)
model_save_dir = './model/' + task_name
os.makedirs(model_save_dir, exist_ok=True)
log_file = task_name + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 全局最小值和最大值
global_min = np.array([0.0, 0.0, 0.0])
global_max = np.array([620.0, 1024.0, 662.0])


def normalize_coordinates(coords, global_min, global_max):
    """
    将坐标归一化到 [0, 1] 范围内。
    """
    return (coords - global_min) / (global_max - global_min)

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

# 自定义排序函数
def natural_sort_key(filename):
    # 使用正则表达式分割字符串，提取数字和非数字部分
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def load_ply(file_path):
    """读取 PLY 文件并返回 (N, 6) 的点云数组"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def save_ply_with_open3d(points, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
    print(f"保存点云到文件 {save_path}")

def _save_ply(points, file_path):
    # 使用open3d保存点云数据为ply格式，不包含颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 只保存坐标信息
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

def merge_blocks(block_folder, output_folder):
    """
    适应情况：文件夹内有多个epoch的block文件进行合并
    """
    blocks = defaultdict(lambda: defaultdict(list))

    # 遍历块文件夹中的所有块文件
    for block_file in sorted(os.listdir(block_folder)):
        if block_file.endswith('.ply'):
            block_file = os.path.basename(block_file)

            parts = block_file.split('_')
            # block_id = parts[4]
            name = '_'.join(parts[:3])
            epoch = str(parts[-1]).replace('epoch','').replace('.ply','')
            block_path = os.path.join(block_folder, block_file)
            blocks[name][epoch].append(block_path)

    # print(blocks)
    # time.sleep(10000)
    # 合并每个特定部分的块
    for name, epoch_files in blocks.items():
        for epoch, block_files in epoch_files.items():
            all_points = []

            for block_file in block_files:
                block_points = load_ply(block_file)
                all_points.append(block_points)

            all_points = np.vstack(all_points)

            output_file = os.path.join(output_folder, f"{name}_eva_merged_epoch{epoch}.ply")

            save_ply_with_open3d(all_points, output_file)
            print(f"合并后的点云保存为: {output_file}")


class PointCloudDataset(Dataset):
    def __init__(self, folder_a, folder_b):
        self.folder_A = folder_a
        self.folder_B = folder_b
        self.file_pairs = self._get_file_pairs()


    def _get_file_pairs(self):
        # 获取folder_A和folder_B中的文件对
        files_A = sorted([f for f in os.listdir(self.folder_A) if f.endswith('.ply')], key=natural_sort_key)
        files_B = sorted([f for f in os.listdir(self.folder_B) if f.endswith('.ply')], key=natural_sort_key)
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
        print(file_A, file_B)
        adjusted_chunk_A = self._load_ply(os.path.join(self.folder_A, file_A))
        adjusted_chunk_B = self._load_ply(os.path.join(self.folder_B, file_B))

        # 转换为张量并返回
        adjusted_chunk_A = torch.tensor(adjusted_chunk_A, dtype=torch.float32)
        adjusted_chunk_B = torch.tensor(adjusted_chunk_B, dtype=torch.float32)

        return adjusted_chunk_A, adjusted_chunk_B




def position_loss(pred, target):
    if isinstance(pred, ME.SparseTensor) and isinstance(target, ME.SparseTensor):
        # 使用稀疏张量的密集特征进行损失计算
        return torch.nn.functional.mse_loss(pred.F, target.F)
    else:
        # 假设 pred 和 target 都是普通张量
        return torch.nn.functional.mse_loss(pred, target)


def train_model(model, data_loader, optimizer, scheduler, device,
                eva_test_file_dir,
                eva_predict_output_dir,
                eva_merged_output_dir,
                epochs=10,
                blocks_per_epoch=8,
                save_epoch_num = 5
                ):
    if os.path.exists(eva_predict_output_dir):
        shutil.rmtree(eva_predict_output_dir)
    if os.path.exists(eva_merged_output_dir):
        shutil.rmtree(eva_merged_output_dir)

    model = model.to(device).float()

    # 用于记录每个epoch的平均损失
    epoch_losses = []
    eva_psnr_results = pd.DataFrame()

    for epoch in range(epochs):
        model.train()
        print(f'开始第{epoch}轮次')
        logging.info(f'开始第{epoch}轮次')
        total_loss = 0
        num_batches = 0
        block_buffer_A, block_buffer_B = [], []

        for (chunks_A, chunks_B) in data_loader:
            if chunks_A.shape[1] > 200:
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

                    # 合并坐标
                    coords_A_batch = np.concatenate(coords_A_batch, axis=0)
                    coords_B_batch = np.concatenate(coords_B_batch, axis=0)
                    print(coords_A_batch.shape)
                    # print(coords_A_batch)
                    # TODO 3 有的block ERROR比较大
                    # for i in range(len(coords_A_batch)):
                    #     error = coords_A_batch[i] - coords_B_batch[i]  # 计算相减误差
                    #     absolute_error = np.abs(error)  # 计算绝对误差
                    #     print(absolute_error)

                    # features需要归一化
                    features_A = torch.tensor(normalize_coordinates(coords_A_batch, global_min, global_max),
                                              dtype=torch.float32).to(device)
                    features_B = torch.tensor(normalize_coordinates(coords_B_batch, global_min, global_max),
                                              dtype=torch.float32).to(device)
                    print(features_A.shape)
                    # print(features_A)

                    # coordinates_A = ME_utils.batched_coordinates([coords_A_batch]).to(device)
                    # coordinates_B = ME_utils.batched_coordinates([coords_B_batch]).to(device)

                    def add_index_column(coords):
                        return torch.cat((torch.zeros(coords.size(0), 1, dtype=torch.float32).to(device), coords),
                                         dim=1)

                    coordinates_A = add_index_column(torch.tensor(coords_A_batch, dtype=torch.float32).to(device))
                    coordinates_B = add_index_column(torch.tensor(coords_B_batch, dtype=torch.float32).to(device))

                    print("Coordinates A shape:", coordinates_A.shape, "Type:", coordinates_A.dtype)
                    print("Coordinates B shape:", coordinates_B.shape, "Type:", coordinates_B.dtype)
                    print("Features A shape:", features_A.shape, "Type:", features_A.dtype)
                    print("Features B shape:", features_B.shape, "Type:", features_B.dtype)

                    origin = ME.SparseTensor(features=features_A, coordinates=coordinates_A)

                    compress = ME.SparseTensor(features=features_B, coordinates=coordinates_B)
                    # print("Coordinate Map Key:", origin.coordinate_map_key)
                    # print("Coordinate Map Key:", compress.coordinate_map_key)

                    # 如果形状不一致，裁剪形状
                    if compress.shape[0] != origin.shape[0]:
                        min_shape = min(compress.shape[0], origin.shape[0])
                        compress = ME.SparseTensor(
                            features=compress.F[:min_shape].float(),
                            coordinates=compress.C[:min_shape],
                            coordinate_manager=compress.coordinate_manager

                        )
                        origin = ME.SparseTensor(
                            features=origin.F[:min_shape].float(),
                            coordinates=origin.C[:min_shape],
                            coordinate_manager=origin.coordinate_manager

                        )

                    # 计算损失
                    output = model(compress)
                    # 计算残差
                    residual = origin.F - compress.F

                    # 计算损失
                    loss = position_loss(output.F.float(), residual.float())
                    # print(output.F.float(), residual.float())
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

        # 加入evaluation
        # 训练集是6结尾的30个ply文件
        # 评估集是7结尾的2个ply文件（切块）
        eva_res = evaluate_model(model, epoch, device,
                       compress_dir = eva_test_file_dir,
                       output_dir = eva_predict_output_dir,
                       merged_output_dir = eva_merged_output_dir,
                       )
        eva_psnr_results = pd.concat([eva_psnr_results, eva_res], ignore_index=True)
        # 每个循环更新一次文件，方便下载查看
        eva_psnr_results.to_excel('eva_psnr_results.xlsx')

        # 每第N个epoch保存模型权重
        if (epoch + 1) % save_epoch_num == 0:
            save_path = './model/' + task_name + '/' + str(epoch + 1) + '_model_residual.pth'
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at epoch {epoch + 1} to {save_path}")
        scheduler.step()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), epoch_losses, label='Training Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Epochs')
        plt.legend()
        plt.savefig('My_Net_Residual_Loss_{}.png'.format(formatted_date))
        plt.close()

    # 训练完成后，绘制损失-epoch图
    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), epoch_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.savefig('My_Net_Residual_Loss_Last_{}.png'.format(formatted_date))
    plt.show()

def predict_on_blocks(model, blocks, global_min, global_max, device):
    model.eval()

    global_min = torch.tensor(global_min, dtype=torch.float32, device=device)
    global_max = torch.tensor(global_max, dtype=torch.float32, device=device)

    coords_B = torch.tensor(blocks[:, :3], dtype=torch.float32).to(device)
    feats_B = torch.tensor(blocks[:, 3:], dtype=torch.float32).to(device)

    normalized_coords_B = normalize_coordinates(coords_B, global_min, global_max)

    # 添加第四维用于批次索引
    batch_index = torch.full((coords_B.shape[0], 1), 0, dtype=torch.int32, device=device)
    int_coords_B = torch.cat([batch_index, coords_B.int()], dim=1)

    print('input_features:',normalized_coords_B)
    print('input_coordinates:',int_coords_B)
    logger.info('Input features: %s', normalized_coords_B)
    logger.info('Input coordinates: %s', int_coords_B)

    inputs = ME.SparseTensor(features=normalized_coords_B, coordinates=int_coords_B)

    with torch.no_grad():  # 禁用梯度计算
        output = model(inputs)
    print('model_predict_res:',output.F)
    logger.info('Model prediction results: %s', output.F)
    denormalize_output = denormalize_coordinates(output.F, global_min, global_max)
    logger.info('Model prediction results after denormalization: %s', denormalize_output)
    logger.info('Shape check: feats_B: %s, inputs: %s, output: %s', feats_B.shape, inputs.shape, output.shape)
    print('model_predict_res_denormalize_coordinates:', denormalize_output)
    print('shape check:', feats_B.shape, inputs.shape, output.shape)

    predicted_coords = normalized_coords_B + output.F
    denormalize_coords = denormalize_coordinates(predicted_coords, global_min, global_max)

    # 使用 detach() 去除梯度信息
    predicted_points = torch.cat((denormalize_coords, feats_B), dim=1).detach().cpu().numpy()

    # 清理未使用的张量
    del coords_B, feats_B, normalized_coords_B, inputs, output, predicted_coords, denormalize_coords
    torch.cuda.empty_cache()

    return predicted_points


def psnr(a_file,b_file):
    # MPEG工具路径
    mpeg_tool_path = "/home/jupyter-eason/data/software/mpeg-pcc-tmc2-master/bin/PccAppMetrics"
    resolution = "1023"
    frame_count = "1"
    # 构建命令
    command = [
        mpeg_tool_path,
        f"--uncompressedDataPath={a_file}",
        f"--reconstructedDataPath={b_file}",
        f"--resolution={resolution}",
        f"--frameCount={frame_count}"
    ]

    # 执行命令并获取输出
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout
    print(output)
    # 使用正则表达式提取 mseF,PSNR (p2point) 的值
    match = re.search(r'mseF,PSNR \(p2point\): ([\d.]+)', output)
    base_a = os.path.basename(a_file)
    base_b = os.path.basename(b_file)
    if match:
        mseF_psnr_value = match.group(1)  # 提取匹配的值
        print(f"提取的值: {mseF_psnr_value}")
    else:
        print("未找到匹配的值。")
        mseF_psnr_value = 0
    print(base_a, base_b, mseF_psnr_value)
    return (base_a, base_b, mseF_psnr_value)


def psnr_pre(epoch, original_path, compress_path, original_kd_path, predict_path):
    original_files =     sorted([ f for f in os.listdir(original_path) if f.endswith('.ply')])
    compress_files =     sorted([ f for f in os.listdir(compress_path) if f.endswith('.ply')])
    original_kd_files =  sorted([ f for f in os.listdir(original_kd_path) if f.endswith('.ply')])
    predict_files  =     sorted([ f for f in os.listdir(predict_path) if f.endswith('.ply')])

    original_files =     [ os.path.join(original_path,f) for f in original_files]
    compress_files =     [ os.path.join(compress_path,f) for f in compress_files]
    original_kd_files =  [ os.path.join(original_kd_path,f) for f in original_kd_files]
    predict_files  =     [ os.path.join(predict_path,f) for f in predict_files]

    # 计算psnr

    # 1.预测与原图比较
    res_1 = [psnr(a, b) for a, b in zip(original_files, predict_files)]
    # 2.预测与压缩比较
    res_2 = [psnr(a, b) for a, b in zip(compress_files, predict_files)]
    # 3.预测与原图kd比较
    res_3 = [psnr(a, b) for a, b in zip(original_kd_files, predict_files)]


    # 4.压缩与原图比较
    res_4 = [psnr(a, b) for a, b in zip(compress_files, original_files)]
    data = {
        "epoch": [epoch] * len(res_1),
        "original_vs_predict": res_1,
        "compress_vs_predict": res_2,
        "original_kd_vs_predict": res_3,
        "compress_vs_original": res_4
    }
    pd_data = pd.DataFrame(data)
    return pd_data

def evaluate_model(model, epoch, device,
                   compress_dir,
                   output_dir,
                   merged_output_dir
                ):

    """
    训练集是6结尾的30个ply文件
    评估集是7结尾的2个ply文件（切块）

    :param epoch:
    :param merged_output_dir:
    :param output_dir:
    :param model:
    :param compress_dir: './test/test2/soldier/block64/adjusted_chunks_B'
    :param device:
    :return:
    """

    # 创建输出文件夹 './test/test2/soldier/evaluate_block/epoch_0'
    output_dir = os.path.join(output_dir, f'epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    # 创建输出文件夹 './test/test2/soldier/evaluate_merged/epoch_0'
    merged_output_dir = os.path.join(merged_output_dir, f'epoch_{epoch}')
    os.makedirs(merged_output_dir, exist_ok=True)

    # 获取 compress_dir 中的所有 .ply 文件
    files_B = sorted([os.path.join(compress_dir, f) for f in os.listdir(compress_dir) if f.endswith('.ply')])

    for file_B in files_B:
        # 提取文件名部分，用于输出文件的命名
        base_name_B = os.path.basename(file_B)
        # 加载并切分文件 B
        points_B = load_ply(file_B)
        # 预测并合并所有块
        predicted_points = predict_on_blocks(model, points_B, global_min, global_max, device)

        # 保存合并后的点云
        base_name_B = base_name_B.replace('.ply', '')
        output_file_path = os.path.join(output_dir, f"{base_name_B}_EvaPredicted_epoch_{str(epoch)}.ply")
        save_ply_with_open3d(predicted_points, output_file_path)

    merge_blocks(block_folder=output_dir, output_folder=merged_output_dir)

    # 此时有4种ply文件
    # 1.original
    # 2.compress （VPCC压缩后）
    # 3.original_kd (compress和original进行kd-tree操作)   它与original的PSNR最佳  70+
    # 4.predict (每个epoch都会预测出来2张，主要是观察PSNR与compress-original的PSNR，能与baseline-loss和real-loss对应上)

    # 2,3,4要分别与1进行PSNR计算；
    # 2,3进行计算
    # 4与2,3进行计算
    # './test/test2/soldier/block64/adjusted_chunks_B'
    base_path = compress_dir.split('block')[0]

    original_path = os.path.join(base_path, 'original')
    compress_path = os.path.join(base_path, 'compress')
    original_kd_path = os.path.join(base_path, 'block64/merged_original')
    # './test/test2/soldier/evaluate_merged/epoch_0'
    predict_path = merged_output_dir

    pd_psnr = psnr_pre(epoch, original_path, compress_path, original_kd_path, predict_path)
    return pd_psnr


def main(mode):
    if mode == 'test_run':
        folder_A = './data_sample/soldier/block/adjusted_chunks_A'
        folder_B = './data_sample/soldier/block/adjusted_chunks_B'

    elif mode == 'full_run':
        folder_A = './data30/soldier/block/adjusted_chunks_A'
        folder_B = './data30/soldier/block/adjusted_chunks_B'
    else:
        raise ValueError("Invalid mode. Use 'test' or 'full'.")

    eva_test_file_dir = './test/test2/soldier/block64/adjusted_chunks_B'
    eva_predict_output_dir = './test/test2/soldier/evaluate_block'
    eva_merged_output_dir = './test/test2/soldier/evaluate_merged'

    dataset = PointCloudDataset(folder_a=folder_A,
                                folder_b=folder_B
                                )

    data_loader = DataLoader(dataset, batch_size=1)

    model = MyNet()
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)  # 使用较小标准差的正态分布初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(weights_init)

    optimizer = torch.optim.Adam([{"params": model.parameters(), 'lr': 0.001}],
                                 betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model = model,
                data_loader = data_loader,
                optimizer = optimizer,
                scheduler = scheduler,
                device=torch.device("cuda:0"),
                eva_test_file_dir=eva_test_file_dir,
                eva_predict_output_dir=eva_predict_output_dir,
                eva_merged_output_dir=eva_merged_output_dir,
                epochs=100,
                blocks_per_epoch=1
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training or testing mode.")
    parser.add_argument("mode", choices=["test_process", "full_process", "test_run", "full_run"],
                        help="Mode to run: 'test' or 'full'")
    args = parser.parse_args()

    main(args.mode)
