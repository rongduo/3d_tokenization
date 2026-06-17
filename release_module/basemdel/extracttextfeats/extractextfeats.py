
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

from transformers import AutoTokenizer, AutoModel

# 文字特征提取网络
class TextFeatureExtractor:
    def __init__(self):
        """
        初始化文字特征提取模块
        :param checkpoint_path: 模型权重路径
        """
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

    def extract_features(self, labels):
        """
        读取点云数据并提取特征
        :param labels: 文字查询列表
        :return: 提取的特征数据
        """
        # with open(f"{file_path}/masks/merged/mask_labels.txt", "r") as f:
        #     labels = f.read().splitlines()
        ## encode label
        inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs) # n_masks, feat_dim (768)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        return text_feat

if __name__ == "__main__":

    # 1. 加载labels
    masklabelpath = '/data4/jl/project/Find3D/dataset/labeled_/rendered/alarm_clock_8f3a450bf7e3414b89ceba7484d074e7/oriented/masks/merged/mask_labels.txt'
    with open(masklabelpath, "r") as f:
        labels = f.read().splitlines()

    # 2. 初始化网络进行测试
    textextr = TextFeatureExtractor()
    textfeats = textextr.extract_features(labels)
    
    print('labels', len(labels))
    print('textfeats:', textfeats.shape)
    # labels 102
    # textfeats: torch.Size([102, 768])