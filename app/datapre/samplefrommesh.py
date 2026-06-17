"""
处理mesh，将模型采样为 点云、颜色和法向量

python -m tmp.samplefrommesh

点云 ： /apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/data/coarse_b29_0cb/points.pt  （对应的ply： points.ply）
对应的模型 ： /apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/data/coarse_b29_0cb/model/model_align.obj
"""

# 2. 在mesh上采样获得点云，颜色和法向量
import os
import torch
import trimesh
import open3d as o3d
import numpy as np
from typing import Tuple

# 核心采样函数（复用原逻辑并简化）
def load_mesh(file_path: str) -> trimesh.Trimesh:
    """加载单个OBJ网格"""
    mesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("无效的网格文件")
    return mesh

def sample_mesh(mesh: trimesh.Trimesh, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """采样点云、颜色、法向量"""
    try:
        points, face_indices, colors = trimesh.sample.sample_surface(mesh, count=num_samples, sample_color=True)
        colors = colors[:, :3] if colors is not None else np.full((num_samples, 3), 0.5)
    except:
        points, face_indices = trimesh.sample.sample_surface(mesh, count=num_samples)
        colors = np.full((num_samples, 3), 0.5)
    
    # 计算法向量 + 归一化颜色
    normals = mesh.face_normals[face_indices]
    colors = colors / 255.0 if colors.max() > 1.0 else colors
    
    # 转换为tensor
    return (
        torch.from_numpy(points).float(),
        torch.from_numpy(colors).float(),
        torch.from_numpy(normals).float()
    )

def save_ply(points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, save_path: str):
    """将点云保存为PLY文件（方便可视化）"""
    # 转换为numpy数组
    points_np = points.numpy()
    colors_np = colors.numpy()
    normals_np = normals.numpy()
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)
    pcd.normals = o3d.utility.Vector3dVector(normals_np)
    
    # 保存PLY文件
    o3d.io.write_point_cloud(save_path, pcd)

# 超参数配置
INPUT_OBJ = "/apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/data/coarse_b29_0cb/model/model_align.obj"
OUTPUT_DIR = "/apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/data/coarse_b29_0cb_sample"
NUM_SAMPLES = 5000  # 采样点数超参数，可按需修改

# 主流程
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载网格 + 采样
    mesh = load_mesh(INPUT_OBJ)
    points, colors, normals = sample_mesh(mesh, NUM_SAMPLES)
    
    # 保存PT文件
    torch.save(points, os.path.join(OUTPUT_DIR, "points.pt"))
    torch.save(colors, os.path.join(OUTPUT_DIR, "rgb.pt"))
    torch.save(normals, os.path.join(OUTPUT_DIR, "normals.pt"))
    
    # 保存PLY文件（方便可视化查看）
    save_ply(points, colors, normals, os.path.join(OUTPUT_DIR, "sampled_point_cloud.ply"))
    
    print(f"采样完成！已保存至 {OUTPUT_DIR}")
    print(f"点云形状: {points.shape}, 颜色形状: {colors.shape}, 法向量形状: {normals.shape}")
    print(f"PLY文件路径: {os.path.join(OUTPUT_DIR, 'sampled_point_cloud.ply')}")



'''
1. 
######## 将点云转为ply文件 ########
import torch
import os
import numpy as np  # 补充numpy导入
from plyfile import PlyData, PlyElement

# ===================== 配置路径 =====================
input_pt_path = "/apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/data/coarse_b29_0cb/points.pt"
output_ply_path = os.path.join(os.path.dirname(input_pt_path), "points.ply")

# ===================== 加载pt文件 =====================
try:
    points = torch.load(input_pt_path, map_location='cpu')
    print(f"成功加载pt文件，点云形状: {points.shape}")
except Exception as e:
    raise ValueError(f"加载pt文件失败: {e}")

# ===================== 数据预处理 =====================
print('points:', points.shape)
if len(points.shape) != 2 or points.shape[1] != 3:
    raise ValueError(f"点云格式错误，需为N×3的张量，当前形状: {points.shape}")

# 转换为numpy数组（float32格式）
points_np = points.numpy().astype('float32')
print('points_np:', points_np.shape)

# ===================== 关键修正：构造一维结构化数组 =====================
# 将二维数组转换为一维结构化数组（每个元素包含x/y/z字段）
points_struct = np.core.records.fromarrays(
    points_np.T,  # 转置后，每一行对应x/y/z的所有坐标值
    names='x,y,z',  # 字段名
    formats='f4,f4,f4'  # 字段类型（float32）
)

# ===================== 保存为PLY文件 =====================
ply_element = PlyElement.describe(points_struct, 'vertex')  # 传入一维结构化数组
PlyData([ply_element], text=True).write(output_ply_path)
print(f"PLY文件已保存至: {output_ply_path}")'''
