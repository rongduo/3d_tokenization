"""
获得3D特征提取模块
输入 : obj文件， 获得（点云、纹理和法向）， 用find3d提取特征
同时，做下特征的可视化
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import open3d as o3d
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors

from model.evaluation.utils import load_model, preprocess_pcd, encode_text
from release_module.obj2ptscolornorm.obj2ptscolornorm import load_and_sample_mesh

class FeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path):
        """
        初始化3D特征提取网络
        :param checkpoint_path: 模型权重路径
        """
        super(FeatureExtractor, self).__init__()
        self.model = load_model(checkpoint_path)  # 加载预训练模型

    def forward(self, xyz, rgb, normal):
        """
        特征提取
        :param xyz: 点云坐标
        :param rgb: 颜色信息
        :param normal: 法向量
        :return: 提取的特征
        """
        # data_dict = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
        # 检查每个输入变量是否为 tensor，如果不是则转换为 tensor
        if not torch.is_tensor(xyz):
            xyz = torch.tensor(xyz, dtype=torch.float32)

        if not torch.is_tensor(rgb):
            rgb = torch.tensor(rgb, dtype=torch.float32)

        if not torch.is_tensor(normal):
            normal = torch.tensor(normal, dtype=torch.float32)
        # 处理点云数据
        data_dict = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
        # text_embeds = encode_text(queries)  # queries 是在主函数指定的
        # data_dict["label_embeds"] = text_embeds

        # debug 保存pts_xyz 点云
        # 假设 pts_xyz 是形状为 (N, 3) 的 torch.Tensor，先转换为 numpy 数组
        pts_np = data_dict['coord'].cpu().numpy()  # 若在GPU上，需先移至CPU
        colors = np.ones_like(pts_np) * 255  # 白色，形状 (N, 3)，值范围 0-255
        colors = colors.astype(np.uint8)
        vertex_data = np.array(
            [(*point, *color) for point, color in zip(pts_np, colors)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        ply_element = PlyElement.describe(vertex_data, "vertex")
        PlyData([ply_element], text=True).write("results_tmp/debug_pts_xyz.ply")
        print("点云已保存为 debug_pts_xyz.ply")
        asdf

        '''# debug : 打印data_dict中数据所在得设备
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} is on device: {value.device}")
        # asdf'''

        with torch.no_grad():
            features = self.model(x=data_dict)  # 网络前向计算
        return features.cpu().numpy(), data_dict['coord'].cpu().numpy()

    def visualize(self, features, xyz, save_path=None):
        """
        使用PCA将特征降维并可视化，或保存为PLY文件
        :param features: 提取的特征
        :param xyz: 点云坐标
        :param save_path: 保存PLY文件的路径; 如果为None，则进行可视化
        """
        # 使用L2范数对特征进行归一化
        data_scaled = features / np.linalg.norm(features, axis=-1, keepdims=True)

        # 使用PCA将特征降维到3维空间
        pca = PCA(n_components=3)
        data_reduced = pca.fit_transform(data_scaled)

        # 将降维后的特征标准化到[0, 1]范围并转换为255的范围
        data_reduced = (data_reduced - data_reduced.min(axis=0)) / (data_reduced.max(axis=0) - data_reduced.min(axis=0))
        colors_255 = (data_reduced * 255).astype(np.uint8)

        assert xyz.shape[0] == colors_255.shape[0], "点云坐标和颜色数量不匹配"

        # 创建点云对象
        points = xyz  # 使用原始点云坐标
        mapped_colors = colors_255  # 使用特征颜色

        # 准备PLY格式数据
        vertex_data = np.array(
            [(*point, *color) for point, color in zip(points, mapped_colors)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        )

        # 创建PLY元素
        el = PlyElement.describe(vertex_data, "vertex")

        # 保存为PLY文件或可视化
        if save_path is not None:
            PlyData([el], text=True).write(save_path)
            print(f"Point cloud saved to: {save_path}")
        else:
            # 否则可视化点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(mapped_colors / 255.0)  # 正规化颜色值
            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")
  

def extract_features_from_obj(obj_path, num_points=5000):
    """
    从obj文件提取点云、颜色和法向量
    :param obj_path: obj文件路径
    :param num_points: 需要采样的点数量
    :return: 点云 (xyz)，颜色 (rgb)，法向 (normal)
    """
    if obj_path.endswith('.obj') or obj_path.endswith('.ply'):
        xyz, rgb, normal = load_and_sample_mesh(obj_path, num_points, colornormflag=True)
    else:
        print("Unsupported file type. Please provide a .obj file.")
        return None, None, None

    return xyz, rgb, normal


from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cosine
# 旋转点云
def rotate_point_cloud(pc):
    # 随机生成一个旋转
    random_rotation = R.random()
    
    # 旋转点云
    rotated_pc = random_rotation.apply(pc)  # 应用旋转
    return rotated_pc

def compute_similarity(feats1, feats2):
    # 计算每对对应特征之间的余弦相似度
    cos_similarities = []
    for i in range(len(feats1)):
        similarity = 1 - cosine(feats1[i], feats2[i])  # 计算第i个特征的相似度
        cos_similarities.append(similarity)
    return np.array(cos_similarities)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract 3D features from an obj file")
    parser.add_argument("--object_path", required=True, type=str, help='The .obj file to extract features from.')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='Path to the model checkpoint.')
    parser.add_argument("--num_points", default=5000, type=int, help='Number of points to sample from the mesh.') # 3664, 5000
    parser.add_argument("--save_path", type=str, default=None, help='Path to save the point cloud as PLY file.')
    args = parser.parse_args()

    ####################旋转前后特征相似度的测试、
    # 提取特征
    xyz, rgb, normal = extract_features_from_obj(args.object_path, args.num_points)
    # 初始化特征提取器
    featsextr = FeatureExtractor(args.checkpoint_path)

    # 提取旋转前的特征
    original_feats, xyz_sub = featsextr.forward(xyz, rgb, normal)
    # 对点云进行旋转
    pc_rotated = rotate_point_cloud(xyz)
    pc_normal = rotate_point_cloud(normal)
    # 提取旋转后的特征
    rotated_feats, rotated_xyz_sub = featsextr.forward(pc_rotated, rgb, pc_normal)
    # 计算相似度
    print('original_feats, rotated_feats : ', original_feats.shape, rotated_feats.shape)
    similarities = compute_similarity(original_feats, rotated_feats)
    avg_similarity = np.mean(similarities)

    print('similarities:', similarities)
    print(f"Average similarity between original and rotated features: {avg_similarity:.4f}")
    asdf



    ####################提取特征和可视化的测试
    # 提取特征
    xyz, rgb, normal = extract_features_from_obj(args.object_path, args.num_points)

    # 检查输入有效性
    if xyz is not None and rgb is not None and normal is not None:
        # 创建特征提取器实例
        feature_extractor = FeatureExtractor(args.checkpoint_path)

        # 特征提取
        features, xyz_sub = feature_extractor.forward(xyz, rgb, normal)

        # 可视化
        feature_extractor.visualize(features, xyz_sub, args.save_path)
