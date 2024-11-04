import os
import re
import subprocess
import pandas as pd

# 定义路径
base_path = "/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/test30"
original_folder = os.path.join(base_path, "original")
compress_folder = os.path.join(base_path, "compress")
model_folders = [f for f in os.listdir(base_path) if f.endswith("_model.pth_output")]

# MPEG工具路径
mpeg_tool_path = "/home/jupyter-eason/data/software/mpeg-pcc-tmc2-master/bin/PccAppMetrics"
reference_path_template = "/home/jupyter-eason/project/upsampling/pc_enhance_simple_script/test30/original/redandblack_vox10_{point}.ply"
resolution = "1023"
frame_count = "1"

# 准备Excel数据列表
results = []

# 匹配并执行命令（模型文件夹与original的对比）
for model_folder in model_folders:
    model_folder_path = os.path.join(base_path, model_folder)
    for model_file in os.listdir(model_folder_path):
        match = re.search(r'_(\d+).ply', model_file)
        if match:
            point_id = match.group(1)
            original_file = f"redandblack_vox10_{point_id}.ply"
            original_file_path = os.path.join(original_folder, original_file)
            model_file_path = os.path.join(model_folder_path, model_file)

            # 构建命令
            command = [
                mpeg_tool_path,
                f"--uncompressedDataPath={reference_path_template.format(point=point_id)}",
                f"--reconstructedDataPath={model_file_path}",
                f"--resolution={resolution}",
                f"--frameCount={frame_count}"
            ]

            # 执行命令并获取输出
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout

            # 获取相对路径
            original_relative = os.path.relpath(original_file_path, original_folder)
            model_output_relative = os.path.relpath(model_file_path, base_path)

            # 将结果添加到列表
            results.append({
                "original": original_relative,
                "model_output": model_output_relative,
                "output": output
            })

# 匹配并执行命令（compress文件夹与original的对比）
for compress_file in os.listdir(compress_folder):
    match = re.search(r'_(\d+).ply', compress_file)
    if match:
        point_id = match.group(1)
        original_file = f"redandblack_vox10_{point_id}.ply"
        original_file_path = os.path.join(original_folder, original_file)
        compress_file_path = os.path.join(compress_folder, compress_file)

        # 构建命令
        command = [
            mpeg_tool_path,
            f"--uncompressedDataPath={reference_path_template.format(point=point_id)}",
            f"--reconstructedDataPath={compress_file_path}",
            f"--resolution={resolution}",
            f"--frameCount={frame_count}"
        ]

        # 执行命令并获取输出
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout

        # 获取相对路径
        original_relative = os.path.relpath(original_file_path, original_folder)
        compress_output_relative = os.path.relpath(compress_file_path, base_path)

        # 将结果添加到列表
        results.append({
            "original": original_relative,
            "compress_output": compress_output_relative,
            "output": output
        })

# 保存到Excel文件
df = pd.DataFrame(results)
df.to_excel("test_results.xlsx", index=False)
print("测试结果已保存到 test_results.xlsx")
