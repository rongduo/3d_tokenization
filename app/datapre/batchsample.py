"""
处理omniobject3d
python -m app.datapre.batchsample
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
        raise ValueError(f"无效的网格文件: {file_path}")
    return mesh

def sample_mesh(mesh: trimesh.Trimesh, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """采样点云、颜色、法向量"""
    try:
        points, face_indices, colors = trimesh.sample.sample_surface(mesh, count=num_samples, sample_color=True)
        colors = colors[:, :3] if colors is not None else np.full((num_samples, 3), 0.5)
    except Exception as e:
        # 捕获具体异常并打印，方便排错
        print(f"采样颜色失败，使用默认灰色: {e}")
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

def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """
    点云规范化：中心化 + scale归一化，将点云映射到[-0.5, 0.5]区间内
    Args:
        points: 原始点云 (N, 3)
    Returns:
        normalized_points: 规范化后的点云 (N, 3)
    """
    # 1. 中心化：减去点云的均值（消除位置偏移）
    center = points.mean(dim=0, keepdim=True)  # (1, 3)
    centered_points = points - center
    
    # 2. 计算最大绝对值（确定点云的最大范围，避免缩放偏差）
    max_abs_value = centered_points.abs().max()
    
    # 防止除以0（极端情况，点云所有点重合）
    if max_abs_value < 1e-8:
        max_abs_value = 1e-8
    
    # 3. scale归一化：映射到[-0.5, 0.5]
    # 除以2*max_abs_value，使得最大绝对值为0.5
    normalized_points = centered_points / (2 * max_abs_value)
    
    return normalized_points

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

def find_all_obj_files(root_dir: str) -> list:
    """
    递归搜索根目录下所有.obj文件
    Args:
        root_dir: 搜索根目录
    Returns:
        obj_file_list: 所有.obj文件的完整路径列表
    """
    obj_file_list = []
    # 递归遍历目录树
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 筛选后缀为.obj的文件（忽略大小写，兼容.OBJ）
            if file.lower().endswith(".obj"):
                obj_file_path = os.path.join(root, file)
                obj_file_list.append(obj_file_path)
    return obj_file_list

# 超参数配置
SEARCH_ROOT_DIR = "/x2robot_v2/lanejin/new_data/dataset/omniobject3d"  # 递归搜索的根目录
NUM_SAMPLES = 5000  # 采样点数超参数，可按需修改

# 主流程
if __name__ == "__main__":
    
    # 1. 递归查找所有.obj文件
    obj_files = find_all_obj_files(SEARCH_ROOT_DIR)
    if not obj_files:
        print(f"在 {SEARCH_ROOT_DIR} 及其子目录下未找到任何.obj文件！")
        exit(1)
    print(f"共找到 {len(obj_files)} 个.obj文件，开始逐个处理...")
    
    # 2. 遍历每个.obj文件进行处理
    for idx, obj_file in enumerate(obj_files, 1):
        try:
            print(f"\n===== 处理第 {idx}/{len(obj_files)} 个文件 =====")
            print(f"当前处理文件: {obj_file}")
            
            # 3. 获取obj文件的同级目录和文件名前缀（用于生成输出文件）
            obj_dir = os.path.dirname(obj_file)  # obj文件的同级目录（保存结果的目录）
            obj_filename = os.path.basename(obj_file)
            obj_prefix = os.path.splitext(obj_filename)[0]  # 去掉.obj后缀的文件名前缀
            
            # 4. 加载网格 + 采样
            mesh = load_mesh(obj_file)
            points, colors, normals = sample_mesh(mesh, NUM_SAMPLES)
            
            # 5. 点云规范化（中心化 + 归一化到[-0.5, 0.5]）
            points_can = normalize_point_cloud(points)
            
            # 6. 定义各输出文件的路径（保存到obj同级目录）
            points_pt_path = os.path.join(obj_dir, f"{obj_prefix}_points.pt")
            rgb_pt_path = os.path.join(obj_dir, f"{obj_prefix}_rgb.pt")
            normals_pt_path = os.path.join(obj_dir, f"{obj_prefix}_normals.pt")
            points_can_pt_path = os.path.join(obj_dir, f"{obj_prefix}_points_can.pt")
            ply_save_path = os.path.join(obj_dir, f"{obj_prefix}_sampled_point_cloud.ply")
            
            # 7. 保存各类PT文件
            torch.save(points, points_pt_path)
            torch.save(colors, rgb_pt_path)
            torch.save(normals, normals_pt_path)
            torch.save(points_can, points_can_pt_path)
            
            # 8. 保存PLY文件（方便可视化查看）
            save_ply(points, colors, normals, ply_save_path)
            
            # 9. 打印当前文件处理结果
            print(f"当前文件处理完成！")
            print(f"  原始点云: {points_pt_path}")
            print(f"  规范化点云: {points_can_pt_path}")
            print(f"  PLY文件: {ply_save_path}")
            print(f"  点云形状: {points.shape}, 规范化点云形状: {points_can.shape}")
        
        except Exception as e:
            print(f"处理文件 {obj_file} 失败！错误信息: {e}")
            continue
    
    print(f"\n===== 所有文件处理完成！=====")



############ 原本的采样缺少规范化
# import os
# import torch
# import trimesh
# import open3d as o3d
# import numpy as np
# from typing import Tuple

# # 核心采样函数（复用原逻辑，无修改）
# def load_mesh(file_path: str) -> trimesh.Trimesh:
#     """加载单个OBJ网格"""
#     mesh = trimesh.load(file_path)
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.to_mesh()
#     if not isinstance(mesh, trimesh.Trimesh):
#         raise ValueError("无效的网格文件")
#     return mesh

# def sample_mesh(mesh: trimesh.Trimesh, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """采样点云、颜色、法向量"""
#     try:
#         points, face_indices, colors = trimesh.sample.sample_surface(mesh, count=num_samples, sample_color=True)
#         colors = colors[:, :3] if colors is not None else np.full((num_samples, 3), 0.5)
#     except:
#         points, face_indices = trimesh.sample.sample_surface(mesh, count=num_samples)
#         colors = np.full((num_samples, 3), 0.5)
    
#     # 计算法向量 + 归一化颜色
#     normals = mesh.face_normals[face_indices]
#     colors = colors / 255.0 if colors.max() > 1.0 else colors
    
#     # 转换为tensor
#     return (
#         torch.from_numpy(points).float(),
#         torch.from_numpy(colors).float(),
#         torch.from_numpy(normals).float()
#     )

# def save_ply(points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, save_path: str):
#     """将点云保存为PLY文件（方便可视化）"""
#     # 转换为numpy数组
#     points_np = points.numpy()
#     colors_np = colors.numpy()
#     normals_np = normals.numpy()
    
#     # 创建Open3D点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_np)
#     pcd.colors = o3d.utility.Vector3dVector(colors_np)
#     pcd.normals = o3d.utility.Vector3dVector(normals_np)
    
#     # 保存PLY文件
#     o3d.io.write_point_cloud(save_path, pcd)

# # 超参数配置（修改为批处理根路径）
# ROOT_DIR = "/x2robot_v2/lanejin/new_data/dataset/omniobject3d"  # 批处理根目录
# NUM_SAMPLES = 5000  # 采样点数超参数，可按需修改
# OBJ_RELATIVE_PATH = "Scan/Scan.obj"  # 每个类别文件夹下OBJ文件的相对路径

# # 批处理主流程
# if __name__ == "__main__":
#     # 遍历根目录下的所有"类别_uid"文件夹（如belt_001）
#     for category_dir_name in os.listdir(ROOT_DIR):
#         # 拼接完整的类别文件夹路径
#         category_dir_path = os.path.join(ROOT_DIR, category_dir_name)
        
#         # 跳过非文件夹类型（确保只处理"类别_uid"文件夹）
#         if not os.path.isdir(category_dir_path):
#             continue
        
#         # 拼接当前类别的OBJ文件完整路径
#         input_obj_path = os.path.join(category_dir_path, OBJ_RELATIVE_PATH)
        
#         # 跳过不存在Scan.obj的文件夹（容错处理）
#         if not os.path.exists(input_obj_path):
#             print(f"警告：未找到OBJ文件，跳过文件夹 {category_dir_path}")
#             continue
        
#         # 定义当前类别的输出路径（直接在类别文件夹下保存结果，保持目录对应）
#         output_dir = category_dir_path  # 结果直接保存到"类别_uid"文件夹下
#         # 若想在类别文件夹下创建单独的采样结果文件夹，可启用下面这行（注释上面一行）
#         # output_dir = os.path.join(category_dir_path, "sampled_result")
        
#         # 创建输出目录（exist_ok=True避免已存在时报错）
#         os.makedirs(output_dir, exist_ok=True)
        
#         try:
#             # 加载网格 + 采样
#             mesh = load_mesh(input_obj_path)
#             points, colors, normals = sample_mesh(mesh, NUM_SAMPLES)
            
#             # 保存PT文件（到对应类别文件夹下）
#             torch.save(points, os.path.join(output_dir, "points.pt"))
#             torch.save(colors, os.path.join(output_dir, "rgb.pt"))
#             torch.save(normals, os.path.join(output_dir, "normals.pt"))
            
#             # 保存PLY文件（方便可视化查看）
#             ply_save_path = os.path.join(output_dir, "sampled_point_cloud.ply")
#             save_ply(points, colors, normals, ply_save_path)
            
#             # 打印当前文件处理完成信息
#             print(f"======================================")
#             print(f"处理完成！类别：{category_dir_name}")
#             print(f"OBJ文件路径: {input_obj_path}")
#             print(f"结果保存路径: {output_dir}")
#             print(f"点云形状: {points.shape}, 颜色形状: {colors.shape}, 法向量形状: {normals.shape}")
#             print(f"PLY文件路径: {ply_save_path}")
        
#         except Exception as e:
#             # 单个文件处理失败不终止整个批处理，打印错误信息继续处理下一个
#             print(f"======================================")
#             print(f"错误：处理文件夹 {category_dir_path} 失败")
#             print(f"错误详情：{str(e)}")
#             continue
    
#     print(f"======================================")
#     print(f"批处理全部执行完毕！根目录：{ROOT_DIR}")