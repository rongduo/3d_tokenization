"""
    测试将点云转为canoncial color
    cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/release_pipeline5ab/ab4_partfieldloss_sizeaug_decoder_canoncolor
    conda activate find3d3
    python pts2canoncolor.py
"""


import torch
import os
import numpy as np

# def normalize_to_rgb(points):
#     """
#     将点云坐标归一化到0-255范围，作为RGB颜色值
#     """
#     # 确保输入是numpy数组
#     if isinstance(points, torch.Tensor):
#         points = points.numpy()
    
#     # 找到每个维度的最小值和最大值
#     min_vals = np.min(points, axis=0)
#     max_vals = np.max(points, axis=0)
    
#     # 计算范围，添加微小值避免除零
#     ranges = max_vals - min_vals
#     ranges[ranges < 1e-8] = 1e-8  # 处理恒值维度
    
#     # 归一化到0-1范围，再转换到0-255并转为整数
#     normalized = (points - min_vals) / ranges
#     rgb = (normalized * 255).astype(np.uint8)
    
#     return rgb


def normalize_to_rgb(points):
    """
    将点云坐标归一化到0-255范围，作为RGB颜色值（纯Tensor实现）
    """
    # 确保输入是torch.Tensor
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    
    # 找到每个维度的最小值和最大值
    min_vals, _ = torch.min(points, dim=0)
    max_vals, _ = torch.max(points, dim=0)
    
    # 计算范围，添加微小值避免除零
    ranges = max_vals - min_vals
    # 处理恒值维度，使用torch.where替代numpy的布尔索引
    ranges = torch.where(ranges < 1e-8, torch.tensor(1e-8, device=points.device), ranges)
    
    # 归一化到0-1范围，再转换到0-255并转为整数
    normalized = (points - min_vals) / ranges
    # 确保值在[0, 1]范围内，防止由于浮点计算误差导致的微小溢出
    normalized = torch.clamp(normalized, 0.0, 1.0)
    rgb = (normalized * 255).to(torch.uint8)
    
    return rgb

def save_point_cloud_as_ply(points, colors, save_path):
    """
    将点云和对应的颜色保存为PLY文件
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 确保输入是numpy数组
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.numpy()
    
    # 打开文件并写入PLY格式数据
    with open(save_path, 'w') as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入每个点的坐标和颜色
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"已成功保存带颜色的点云到: {save_path}")

def main():
    # 点云输入路径
    # input_path = "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest/airplane/coarse_b'00_01d'/points.pt"
    input_path = "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest/chair/coarse_b'0c_426'/points.pt"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在 - {input_path}")
        return
    
    # 读取点云数据
    try:
        points = torch.load(input_path)
        print(f"成功读取点云数据，共 {len(points)} 个点")
    except Exception as e:
        print(f"读取点云数据时出错: {str(e)}")
        return
    
    # 确保点云是N×3的形状
    if points.ndim != 2 or points.shape[1] != 3:
        print(f"错误: 点云数据格式不正确，期望形状为 (N, 3)，实际形状为 {points.shape}")
        return
    
    # 将x, y, z坐标转换为RGB颜色
    colors = normalize_to_rgb(points).numpy()
    
    # 保存路径
    save_dir = "/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/pts2canonicalcolor"
    # 从输入文件名提取保存文件名
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    save_path = os.path.join(save_dir, f"{input_filename}_colored3.ply")
    
    # 保存为PLY文件
    try:
        save_point_cloud_as_ply(points, colors, save_path)
    except Exception as e:
        print(f"保存PLY文件时出错: {str(e)}")

if __name__ == "__main__":
    main()
