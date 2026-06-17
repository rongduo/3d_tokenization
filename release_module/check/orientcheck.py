"""

    确定训练数据，是否都在规范空间中
    python -m release_module.check.orientcheck

"""
####### 确定训练数据，是否都在规范空间中
import torch
import os
import open3d as o3d
import numpy as np

# 配置路径
data_root = '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/bottle'
save_dir = 'results_tmp/oracheck'
os.makedirs(save_dir, exist_ok=True)

def get_direction_vector(pts_xyz, lid_mask):
    """
    计算lid的方向向量：lid平均点 减去 整体点云平均点
    返回归一化后的方向向量
    """
    # 整体点云的平均点
    all_mean = pts_xyz.mean(dim=0)
    # lid点的平均点
    lid_pts = pts_xyz[lid_mask]
    lid_mean = lid_pts.mean(dim=0)
    # 方向向量（lid平均点 - 整体平均点）
    direction = lid_mean - all_mean
    # 归一化
    direction_norm = direction / (torch.norm(direction) + 1e-8)
    return direction_norm

def is_same_direction(vec1, vec2, threshold=0.9):
    """判断两个方向向量是否同向（余弦相似度 >= threshold）"""
    cos_sim = torch.dot(vec1, vec2)
    return cos_sim >= threshold

def save_point_cloud(xyz, rgb, save_path):
    """保存点云为PLY文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgb.numpy() / 255.0)  # 颜色归一化到0-1
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"已保存点云至: {save_path}")

# 读取物体路径列表
with open(f"{data_root}/train.txt", "r") as f:
    obj_path_list = [line.strip() for line in f if line.strip()]

if not obj_path_list:
    raise ValueError("train.txt中未找到有效路径")

# 存储结果的列表
same_direction = []  # 朝向一致的物体 (data_dir, xyz, rgb, direction)
diff_direction = []   # 朝向不一致的物体 (data_dir, xyz, rgb, direction)
first_direction = None  # 基准方向向量
first_xyz = None
first_rgb = None

# 遍历所有物体，计算方向并分类
for idx, data_dir in enumerate(obj_path_list):
    # 检查必要文件
    required_files = [
        "mask2points.pt", "normals_transformed.pt", 
        "mask_labels.txt", "points_transformed.pt", "rgb.pt"
    ]
    file_paths = [os.path.join(data_dir, f) for f in required_files]
    if not all(os.path.exists(p) for p in file_paths):
        print(f"警告：{data_dir} 缺少文件，跳过")
        continue

    # 加载数据
    with open(file_paths[2], "r") as f:
        labels = f.read().splitlines()
    mask_pts = torch.load(file_paths[0], map_location="cpu")  # [num_labels, num_points]
    normal = torch.load(file_paths[1], map_location="cpu")     # [num_points, 3]
    pts_xyz = torch.load(file_paths[3], map_location="cpu")    # [num_points, 3]
    pts_rgb = torch.load(file_paths[4], map_location="cpu") * 255  # [num_points, 3]

    # 检查lid标签
    if "lid" not in labels:
        print(f"警告：{data_dir} 无lid标签，跳过")
        continue
    lid_idx = labels.index("lid")
    lid_mask = mask_pts[lid_idx] == 1  # lid对应的点掩码

    # 检查lid点是否有效
    if lid_mask.sum() == 0:
        print(f"警告：{data_dir} 无有效lid点，跳过")
        continue

    # 计算方向向量（lid平均点 相对 整体平均点的方向）
    direction = get_direction_vector(pts_xyz, lid_mask)

    # 处理第一个物体（基准）
    if idx == 0:
        first_direction = direction
        first_xyz = pts_xyz
        first_rgb = pts_rgb
        print(f"已设置基准物体：{data_dir}")
        continue

    # 判断与基准方向是否一致
    same = is_same_direction(direction, first_direction)
    if same:
        same_direction.append((data_dir, pts_xyz, pts_rgb, direction))
        print(f"物体 {data_dir} 与基准朝向一致（累计：{len(same_direction)}）")
    else:
        diff_direction.append((data_dir, pts_xyz, pts_rgb, direction))
        print(f"物体 {data_dir} 与基准朝向不一致（累计：{len(diff_direction)}）")

# 计算比例
total = len(same_direction) + len(diff_direction)
same_ratio = len(same_direction) / total if total > 0 else 0.0
diff_ratio = len(diff_direction) / total if total > 0 else 0.0

# 保存点云（基准+最多2个同向+最多2个不同向）
saved = 0
# 保存基准物体
if first_xyz is not None:
    save_path = os.path.join(save_dir, "0_baseline.ply")
    save_point_cloud(first_xyz, first_rgb, save_path)
    saved += 1

# 保存同向物体
save_same = min(2, len(same_direction))
for i in range(save_same):
    data_dir, xyz, rgb, _ = same_direction[i]
    name = os.path.basename(data_dir)
    save_path = os.path.join(save_dir, f"same_{i+1}_{name}.ply")
    save_point_cloud(xyz, rgb, save_path)
    saved += 1

# 保存不同向物体
save_diff = min(2, len(diff_direction))
for i in range(save_diff):
    data_dir, xyz, rgb, _ = diff_direction[i]
    name = os.path.basename(data_dir)
    save_path = os.path.join(save_dir, f"diff_{i+1}_{name}.ply")
    save_point_cloud(xyz, rgb, save_path)
    saved += 1

# 输出统计结果
print("\n" + "="*50)
print(f"基准物体方向向量：{first_direction.numpy()}")
print(f"总有效物体数：{total}（基准物体除外）")
print(f"同向物体数：{len(same_direction)}，占比：{same_ratio:.2%}")
print(f"不同向物体数：{len(diff_direction)}，占比：{diff_ratio:.2%}")
print(f"已保存 {saved} 个点云文件至 {save_dir}")
print(f" - 基准物体：1个")
print(f" - 同向物体：{save_same}个（共{len(same_direction)}个）")
print(f" - 不同向物体：{save_diff}个（共{len(diff_direction)}个）")
print("="*50)
