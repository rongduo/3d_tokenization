"""
 实现功能 ： 输入.obj文件，输出顶点、颜色和法向
 # obj_path = '/data3/jl/mesh_primitive_fitting/data/cat0.9/models/model_normalized.obj'  # 替换为您的 OBJ 文件路径
"""

import trimesh
import torch
import numpy as np
import open3d as o3d

def load_and_sample_mesh(obj_file_path, num_samples, colornormflag=False):
    # 加载OBJ文件
    mesh = trimesh.load(obj_file_path)

    # 确保网格有效
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded mesh is not a valid Trimesh object.")

    # 从网格上采样点
    points, face_indices, colors = trimesh.sample.sample_surface(mesh, count=num_samples, sample_color=True)

    # 计算法向量
    normals = mesh.face_normals[face_indices]  # 从面法线中获取对应的法向量

    # 转换为tensor（可选）
    points_tensor = torch.as_tensor(points, dtype=torch.float32)
    colors_tensor = torch.as_tensor(colors, dtype=torch.float32)
    normals_tensor = torch.as_tensor(normals, dtype=torch.float32)
    if colornormflag:
        # 把颜色变到区间 [0, 1]
        colors_tensor = colors_tensor / 255.0
        return points_tensor, colors_tensor[:,:3], normals_tensor
    else:
        return points_tensor, colors_tensor, normals_tensor


def visualize_with_open3d(points, colors, normals):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 填充点云数据
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 颜色转换为正确的范围并填充
    # colors = (colors * 255).astype(np.uint8)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)  # 确保范围在[0, 1]

    # 创建法向量的箭头
    arrow_scale = 0.1  # 缩放因子
    arrows = []  # 存储箭头
    for point, normal in zip(points, normals):
        arrow_start = point
        arrow_end = point + normal * arrow_scale
        
        # 创建箭头并添加到箭头列表
        arrow = o3d.geometry.LineSet()
        arrow.points = o3d.utility.Vector3dVector([arrow_start, arrow_end])
        arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow.paint_uniform_color([1, 0, 0])  # 设置箭头颜色为红色
        arrows.append(arrow)

    # 可视化点云和法向量
    o3d.visualization.draw_geometries([pcd] + arrows, window_name="Point Cloud with Normals")

if __name__ == "__main__":
    # 示例用法
    obj_path = '/data3/jl/mesh_primitive_fitting/data/cat0.9/models/model_normalized.obj'   # 替换为您的OBJ文件路径
    num_samples = 100000  # 期望采样的点数
    points, colors, normals = load_and_sample_mesh(obj_path, num_samples)

    # 打印结果
    print("Points:\n", points)  # n * 3
    print("Colors:\n", colors) # n * 4  RGBA  [236., 219., 204., 255.],
    print("Normals:\n", normals) # n * 3


    # 可视化
    visualize_with_open3d(points.numpy(), colors.numpy(), normals.numpy())
