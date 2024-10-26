import os
import re  # 用于从文件名中提取编号
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

def load_model_for_prediction(model, model_path='model.pth', device='cuda'):
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def predict(model, data_loader, device='cuda'):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for (chunks_A, chunks_B) in data_loader:
            block_buffer_A, block_buffer_B = [], []
            block_buffer_A.extend(chunks_A)
            block_buffer_B.extend(chunks_B)

            # 仅处理第一个块用于示例
            coords_A1, feats_A1 = block_buffer_A[0][0, :, :3], block_buffer_A[0][0, :, 3:]
            coords_B1, feats_B1 = block_buffer_B[0][0, :, :3], block_buffer_B[0][0, :, 3:]

            coords_A_tensor = ME_utils.batched_coordinates([coords_A1.view(-1, 3)], device=device)
            feats_A_tensor = feats_A1.to(device).float()

            coords_B_tensor = ME_utils.batched_coordinates([coords_B1.view(-1, 3)], device=device)
            feats_B_tensor = feats_B1.to(device).float()

            origin = ME.SparseTensor(features=feats_A_tensor, coordinates=coords_A_tensor)
            x = ME.SparseTensor(features=feats_B_tensor, coordinates=coords_B_tensor)

            new_x = pad_sparse_tensor(x, origin)
            new_x = ME.SparseTensor(features=new_x.F.float(), coordinates=new_x.C,
                                    coordinate_manager=new_x.coordinate_manager)

            output = model(new_x)
            print("Prediction output shape:", output.F.shape)
            return output


# 加载训练好的模型并进行预测
model = MyTestNet()
model = model.float()

# 加载模型权重
model = load_model_for_prediction(model, model_path='model.pth')

# 进行预测
output = predict(model, data_loader)
