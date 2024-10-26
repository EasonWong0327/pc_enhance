import numpy as np
import open3d as o3d
import os

def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def save_ply(points, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)

def get_blocks(points, block_size, cube_size):
    blocks = []
    for x in range(0, cube_size, block_size):
        for y in range(0, cube_size, block_size):
            for z in range(0, cube_size, block_size):
                block_points = points[
                    (points[:, 0] >= x) & (points[:, 0] < x + block_size) &
                    (points[:, 1] >= y) & (points[:, 1] < y + block_size) &
                    (points[:, 2] >= z) & (points[:, 2] < z + block_size)
                ]
                if len(block_points) > 0:
                    blocks.append((block_points, (x, y, z)))
    return blocks

def filter_blocks(blocks, threshold):
    filtered_blocks = []
    for block_points, block_index in blocks:
        if len(block_points) >= threshold:
            filtered_blocks.append((block_points, block_index))
    return filtered_blocks

def main(origin_dir, compress_dir, output_dir, block_size=256, cube_size=1024, threshold=10):
    origin_files = sorted([f for f in os.listdir(origin_dir) if f.endswith('.ply')])
    compress_files = sorted([f for f in os.listdir(compress_dir) if f.endswith('.ply')])

    for origin_file, compress_file in zip(origin_files, compress_files):
        origin_points = load_ply(os.path.join(origin_dir, origin_file))
        compress_points = load_ply(os.path.join(compress_dir, compress_file))

        origin_blocks = get_blocks(origin_points, block_size, cube_size)
        compress_blocks = get_blocks(compress_points, block_size, cube_size)

        origin_blocks = filter_blocks(origin_blocks, threshold)

        for (origin_block, origin_index), (compress_block, compress_index) in zip(origin_blocks, compress_blocks):
            if origin_index == compress_index:
                origin_output_path = os.path.join(output_dir, f'origin_{origin_file[:-4]}_{origin_index}.ply')
                compress_output_path = os.path.join(output_dir, f'compress_{compress_file[:-4]}_{compress_index}.ply')
                save_ply(origin_block, origin_output_path)
                save_ply(compress_block, compress_output_path)

if __name__ == "__main__":
    origin_dir = 'data/original'
    compress_dir = 'data/compress'
    output_dir = 'data/output'
    main(origin_dir, compress_dir, output_dir)
