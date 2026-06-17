"""
 输入数据集中，待测试物体的dir 路径 ； 输入find3d 的预训练模型 ； 保存特征 ， 和分割结果
 注意 ： 在find3d 目录下执行 ； 执行代码为 ： python -m release_module.check.featsandseg
"""
import os
import open3d as o3d
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from release_module.basemdel.extract3dfeats_find3d.ex3dfeats import  extract_features_from_obj, FeatureExtractor

# 1. 首先找到partnet测试的前5个chair的路径
patnet_chairs = ['/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test/test/Chair/179/pc.ply', \
                    '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test/test/Chair/2230/pc.ply', \
                    '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test/test/Chair/2320/pc.ply', \
                    '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test/test/Chair/2364/pc.ply', \
                    '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test/test/Chair/2440/pc.ply']

# 2. 指定特定的预训练模型
# checkpoint_path = 'model/checkpoints/ckpt_80.pth'
# save_dir = '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/featscheck/orgmodel'
# checkpoint_path = 'results/find3d_d3compat/ckpt_200.pth'
# save_dir = '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/featscheck/3dcompat'
checkpoint_path = 'results/find3d_d3compat_prosegloss/ckpt_200.pth'
save_dir = '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/featscheck/3dcompatprosegloss'
os.makedirs(save_dir, exist_ok=True)
num_points = 5000
feature_extractor = FeatureExtractor(checkpoint_path)  # # 创建特征提取器实例

for i, object_path in enumerate(patnet_chairs):
    # 3. 获得对应物体特征的可视化
    save_path = os.path.join(save_dir, f'chair_{i}.ply')
    # 获得ply文件的点云，颜色，法向量
    pcd = o3d.io.read_point_cloud(object_path)
    pts_xyz = torch.tensor(np.asarray(pcd.points)).float()

    '''   # debug 保存pts_xyz 点云
    # 假设 pts_xyz 是形状为 (N, 3) 的 torch.Tensor，先转换为 numpy 数组
    pts_np = pts_xyz.cpu().numpy()  # 若在GPU上，需先移至CPU
    colors = np.ones_like(pts_np) * 255  # 白色，形状 (N, 3)，值范围 0-255
    colors = colors.astype(np.uint8)
    vertex_data = np.array(
        [(*point, *color) for point, color in zip(pts_np, colors)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    ply_element = PlyElement.describe(vertex_data, "vertex")
    PlyData([ply_element], text=True).write("results_tmp/debug_pts_xyz.ply")
    print("点云已保存为 debug_pts_xyz.ply")
    asdf'''



    pts_rgb = torch.tensor(np.asarray(pcd.colors))#*255
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=300))
    normal = torch.tensor(np.asarray(pcd.normals)).float()
    # normalize
    # this is the same preprocessing done before training
    center = pts_xyz.mean(0)
    scale = max((pts_xyz - center).abs().max(0)[0])
    pts_xyz -= center
    pts_xyz *= (0.75 / float(scale)) # put in 1.5-size box
    random_indices = torch.randint(0, pts_xyz.shape[0], (5000,))
    xyz = pts_xyz[random_indices].float()
    rgb = pts_rgb[random_indices].float()
    normal = normal[random_indices].float()

    # debug 把 xyz， rgb， normal进行保存
    import json
    with open(f"/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test/PartNetE_meta.json") as f:
            all_mapping = json.load(f)
    part_names = all_mapping["Chair"]
    debugsavedir = f'/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/featscheck/test/{object_path.split("/")[-2]}'
    os.makedirs(debugsavedir, exist_ok=True)
    # mask_labels.txt \ normals.pt \ points.pt \ rgb.pt
    with open(f"{debugsavedir}/mask_labels.txt", "w") as f:
        for name in part_names:
            f.write(f"{name}\n")
    torch.save(normal, f"{debugsavedir}/normals.pt")
    torch.save(xyz, f"{debugsavedir}/points.pt")
    torch.save(rgb, f"{debugsavedir}/rgb.pt")
    print(f"Saved debug files to {debugsavedir}")


    if rgb.max()>1:
        rgb = rgb
    # 提取特征冰河可视化
    if xyz is not None and rgb is not None and normal is not None:  # 检查输入有效性
        features, xyz_sub = feature_extractor.forward(xyz, rgb, normal)  # # 特征提取
        feature_extractor.visualize(features, xyz_sub, save_path) # 可视化


# --------------------------------

# 1. 然后找到3dcompat中参与训练的chair的dir 路径 ； 进而输出分割结果，看是否过拟合
