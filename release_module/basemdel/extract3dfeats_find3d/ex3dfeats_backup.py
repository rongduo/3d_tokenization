"""
获得3D特征提取模块
输入 : obj文件， 获得（点云、纹理和法向）， 用find3d提取特征
同时，做下特征得可视化
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

# 从obj文件中获得信息
def extract_info_from_obj(obj_path, num_points=5000):
    """
    从obj文件提取点云、颜色和法向量
    :param obj_path: obj文件路径
    :param num_points: 需要采样的点数量
    :return: 点云 (xyz)，颜色 (rgb)，法向 (normal)
    """
    if obj_path.endswith('.obj'):
        xyz, rgb, normal = load_and_sample_mesh(obj_path, num_points, colornormflag=True)
    else:
        print("Unsupported file type. Please provide a .obj file.")
        return None, None, None

    return xyz, rgb, normal


# 3D特征提取网络
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
        data_dict = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
        # text_embeds = encode_text(queries)  # queries 是在主函数指定的
        # data_dict["label_embeds"] = text_embeds

        '''# debug : 打印data_dict中数据所在得设备
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} is on device: {value.device}")
        # asdf'''

        with torch.no_grad():
            features = self.model(x=data_dict)  # 网络前向计算
        return features.cpu().numpy()


# 文字特征提取网络
class TextFeatureExtractor:
    def __init__(self):
        """
        初始化文字特征提取模块
        :param checkpoint_path: 模型权重路径
        """
        a = 1

    def extract_features(self, queries):
        """
        读取点云数据并提取特征
        :param queries: 文字查询列表
        :return: 提取的特征数据
        """
        textfeats = encode_text(queries)  # 将文字查询编码
        return textfeats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract 3D features from an obj file")
    parser.add_argument("--object_path", required=True, type=str, help='The .obj file to extract features from.')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='Path to the model checkpoint.')
    parser.add_argument("--num_points", default=5000, type=int, help='Number of points to sample from the mesh.') # 3664, 5000
    parser.add_argument("--save_path", type=str, default=None, help='Path to save the point cloud as PLY file.')
    args = parser.parse_args()

    # 提取3D语义特征
    xyz, rgb, normal = extract_info_from_obj(args.object_path, args.num_points)
    if xyz is not None and rgb is not None and normal is not None: # 检查输入有效性
        feature_extractor = FeatureExtractor(args.checkpoint_path) # # 创建特征提取器实例
        features = feature_extractor.forward(xyz, rgb, normal)  # # 特征提取
    
    # 提取文字特征
    queries = ["head", "ear", "a sofa"]  # 示例查询

