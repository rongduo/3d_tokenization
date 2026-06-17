import torch
from types import SimpleNamespace
from model.backbone.pt3.model import PointSemSeg
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel
import open3d as o3d
from common.utils import visualize_pts


import torch
from types import SimpleNamespace
# 按需导入所有网络类（或用importlib动态导入，避免硬编码过多import）
from model.backbone.pt3.model import PointSemSeg
# from release_pipeline5ab.ab3_partfieldloss_sizeaug_decoder.mixdecodernet import PointSemSegWithDecoder as PointSemSegWithDecoder_ab3
# from release_pipeline3.stage1_semanspace.halfd3com_worot_aligncates_decoder.mixdecodernet import PointSemSegWithDecoder
# from release_pipeline5ab.ab4_partfieldloss_sizeaug_decoder_canoncolor.mixdecodernet import PointSemSegWithDecoder as PointSemSegWithDecoder_ab4
# from release_pipeline5ab.ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign.mixdecodernet import PointSemSegWithDecoder as PointSemSegWithDecoder_ab5
# from release_pipeline5ab.ab6_partfieldloss_sizeaug_decoder_bbox.mixdecodernet import PointSemSegWithbboxDecoder
# def muti_load_model(
#     checkpoint_path: str,
#     net_type: str = "net1",  # 超参数：指定网络类型，默认net1
#     dim_output: int = 768,  # 原始函数的dim_output，可按需调整
#     extra_model_kwargs: dict = None  # 额外网络参数（如decoder_dim、bbox_head_dim）
# ) -> torch.nn.Module:
#     """
#     加载6个点云语义分割网络的统一函数
#     Args:
#         checkpoint_path: 预训练模型权重路径（含model_state_dict的字典）
#         net_type: 网络类型标识，可选值：
#                  - "net1": model.backbone.pt3.model.PointSemSeg（原网络1）
#                  - "net2": model.backbone.pt3.model.PointSemSeg（原网络2，同net1结构）
#                  - "net3": ab3分支的PointSemSegWithDecoder
#                  - "net4": ab4分支的PointSemSegWithDecoder
#                  - "net5": ab5分支的PointSemSegWithDecoder
#                  - "net6": ab6分支的PointSemSegWithbboxDecoder
#         dim_output: 网络输出维度（原函数参数，适配PointSemSeg的dim_output）
#         extra_model_kwargs: 额外网络参数（如decoder_dim、bbox_head_dim），默认None
#     Returns:
#         加载权重后的eval模式网络（已移至cuda）
#     """
#     # 初始化默认额外参数（避免None报错）
#     extra_model_kwargs = extra_model_kwargs if extra_model_kwargs is not None else {}
    
#     # 1. 创建args（保留原函数的SimpleNamespace逻辑，按需扩展参数）
#     args = SimpleNamespace()
#     # 可根据net_type补充args的专属参数（如部分网络需要的num_classes、in_channels）
#     args.num_classes = extra_model_kwargs.get("num_classes", 10)  # 示例：默认10类语义分割
#     args.in_channels = extra_model_kwargs.get("in_channels", 3)    # 示例：默认3维点云特征（xyz）

#     # 2. 根据net_type初始化对应网络
#     if net_type in ["net1", "net2"]:
#         # 网络1/2：PointSemSeg（仅需dim_output参数）
#         model = PointSemSeg(args=args, dim_output=dim_output)
#         model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    
#     elif net_type == "net3":
#         # 网络3：ab3分支的PointSemSegWithDecoder（需补充decoder参数）
#         model = PointSemSegWithDecoder(args=args)
#         model.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"])
    
#     elif net_type == "net4":
#         # 网络4：ab4分支的PointSemSegWithDecoder（同ab3结构，仅路径不同）
#         model = PointSemSegWithDecoder_ab4(
#             args=args,
#             dim_output=dim_output,
#             decoder_dim=extra_model_kwargs.get("decoder_dim", 256)
#         )
    
#     elif net_type == "net5":
#         # 网络5：ab5分支的PointSemSegWithDecoder（同ab3/ab4结构）
#         model = PointSemSegWithDecoder_ab5(
#             args=args,
#             dim_output=dim_output,
#             decoder_dim=extra_model_kwargs.get("decoder_dim", 256)
#         )
    
#     elif net_type == "net6":
#         # 网络6：带bbox decoder的PointSemSegWithbboxDecoder（需补充bbox参数）
#         model = PointSemSegWithbboxDecoder(
#             args=args,
#             dim_output=dim_output,
#             decoder_dim=extra_model_kwargs.get("decoder_dim", 256),
#             bbox_head_dim=extra_model_kwargs.get("bbox_head_dim", 128)  # bbox分支专属参数
#         )
    
#     else:
#         raise ValueError(
#             f"无效的net_type：{net_type}，可选值为['net1','net2','net3','net4','net5','net6']"
#         )

#     # # 3. 加载预训练权重（兼容单GPU/多GPU权重）
#     # checkpoint = torch.load(checkpoint_path, map_location="cpu")
#     # # 处理多GPU训练的权重（去除"module."前缀）
#     # if "model_state_dict" in checkpoint:
#     #     state_dict = checkpoint["model_state_dict"]
#     # else:
#     #     # 兼容直接保存state_dict的情况（若权重文件仅存参数，无model_state_dict键）
#     #     state_dict = checkpoint
    
#     # if "module." in list(state_dict.keys())[0]:
#     #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
#     # # 加载权重（strict=False：允许部分参数不匹配，如新增的decoder/bbox分支）
#     # model.load_state_dict(state_dict, strict=False)

#     # 4. 切换eval模式并移至cuda（保留原函数逻辑）
#     model.eval()
#     if torch.cuda.is_available():
#         model = model.cuda()
#     else:
#         print("警告：未检测到CUDA，模型将在CPU上运行")

#     return model
    
# def load_model(checkpoint_path):
#     args = SimpleNamespace()
#     model = PointSemSeg(args=args, dim_output=768)
#     model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
#     model.eval()
#     model = model.cuda()
#     return model

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def grid_sample_numpy(xyz, rgb, normal, grid_size): # this should hopefully be 5000 or close
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    normal = normal.cpu().numpy()

    scaled_coord = xyz / np.array(grid_size)
    grid_coord = np.floor(scaled_coord).astype(int)
    min_coord = grid_coord.min(0)
    grid_coord -= min_coord
    scaled_coord -= min_coord
    min_coord = min_coord * np.array(grid_size)
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[0:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]

    grid_coord = grid_coord[idx_unique]
    
    xyz = torch.tensor(xyz[idx_unique]).cuda()
    rgb = torch.tensor(rgb[idx_unique]).cuda()
    normal = torch.tensor(normal[idx_unique]).cuda()
    grid_coord = torch.tensor(grid_coord).cuda()

    return xyz, rgb, normal, grid_coord
    

def preprocess_pcd(xyz, rgb, normal): # rgb should be 0-1
    assert rgb.max() <=1
    # normalize
    # this is the same preprocessing I do before training
    center = xyz.mean(0)
    scale = max((xyz - center).abs().max(0)[0])
    xyz -= center
    xyz *= (0.75 / float(scale)) # put in 0.75-size box

    # axis swap
    xyz = torch.cat([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], dim=1)

    # center shift
    xyz_min = xyz.min(dim=0)[0]
    xyz_max = xyz.max(dim=0)[0]
    xyz_max[2] = 0
    shift = (xyz_min+xyz_max)/2
    xyz -= shift

    # subsample/upsample to 5000 pts for grid sampling
    if xyz.shape[0] != 5000:
        random_indices = torch.randint(0, xyz.shape[0], (5000,))
        pts_xyz_subsampled = xyz[random_indices]
        pts_rgb_subsampled = rgb[random_indices]
        normal_subsampled = normal[random_indices]
    else:
        pts_xyz_subsampled = xyz
        pts_rgb_subsampled = rgb
        normal_subsampled = normal

    # grid sampling
    pts_xyz_gridsampled, pts_rgb_gridsampled, normal_gridsampled, grid_coord = grid_sample_numpy(pts_xyz_subsampled, pts_rgb_subsampled, normal_subsampled, 0.02)

    # another center shift, z=false
    xyz_min = pts_xyz_gridsampled.min(dim=0)[0]
    xyz_min[2] = 0
    xyz_max = pts_xyz_gridsampled.max(dim=0)[0]
    xyz_max[2] = 0
    shift = (xyz_min+xyz_max)/2
    pts_xyz_gridsampled -= shift
    xyz -= shift

    # normalize color
    pts_rgb_gridsampled = pts_rgb_gridsampled / 0.5 - 1

    # combine color and normal as feat
    feat = torch.cat([pts_rgb_gridsampled, normal_gridsampled], dim=1)

    data_dict = {}
    data_dict["coord"] = pts_xyz_gridsampled
    data_dict["feat"] = feat
    data_dict["grid_coord"] = grid_coord
    data_dict["xyz_full"] = xyz
    data_dict["offset"] = torch.tensor([pts_xyz_gridsampled.shape[0]]).to(pts_xyz_gridsampled.device)
    return data_dict


def encode_text(texts):
    siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")
    inputs = tokenizer(texts, padding="max_length", return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].cuda()
    with torch.no_grad():
        text_feat = siglip.cuda().get_text_features(**inputs)
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    return text_feat

def read_ply(obj_path, visualize=True):
    pcd = o3d.io.read_point_cloud(obj_path)
    if visualize:
        visualize_pts(torch.tensor(np.asarray(pcd.points)), torch.tensor(np.asarray(pcd.colors)), save_path="actual")
    xyz = torch.tensor(np.asarray(pcd.points)).float()
    rgb = torch.tensor(np.asarray(pcd.colors)).float()
    normal = torch.tensor(np.asarray(pcd.normals)).float()
    return xyz, rgb, normal


def read_pcd(obj_path, visualize=True):
    pcd = o3d.io.read_point_cloud(obj_path)
    if visualize:
        visualize_pts(torch.tensor(np.asarray(pcd.points)), torch.tensor(np.asarray(pcd.colors)), save_path="actual")
    xyz = torch.tensor(np.asarray(pcd.points)).float()
    rgb = torch.tensor(np.asarray(pcd.colors)).float()
    normal = torch.tensor(np.asarray(pcd.normals)).float()
    return xyz, rgb, normal








####################
import os
import torch
import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn as nn

def process_embeddings(shape_embeds: torch.Tensor, text_embeds: torch.Tensor, n_clusters: int) -> Tuple[torch.Tensor, dict]:
    """
    处理嵌入向量，执行聚类、相似度计算并生成类别标签
    
    参数:
        shape_embeds: 形状嵌入向量
        text_embeds: 文本嵌入向量
        n_clusters: 聚类数量（超参数）
    
    返回:
        labels: 每个点的类别标签，从1开始，未分类的为0
        info: 包含中间结果的字典，如相似度、聚类中心等
    """
    # 确保输入是正确的张量类型并移动到CPU进行处理
    if isinstance(shape_embeds, torch.Tensor):
        shape_embeds_np = shape_embeds.cpu().numpy()
    else:
        shape_embeds_np = np.array(shape_embeds)
    
    # 1. 执行KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(shape_embeds_np)
    cluster_labels = kmeans.labels_  # 聚类标签从0开始
    
    # 2. 计算每个聚类与文本的相似度
    part_similarities = []
    for cluster_id in range(n_clusters):
        # 获取该聚类的所有点
        mask = (cluster_labels == cluster_id)
        if np.sum(mask) == 0:
            part_similarities.append((cluster_id, -1.0))  # 空聚类相似度设为最低
            continue
        
        # 计算聚类的平均嵌入
        cluster_embeds = shape_embeds[mask]
        cluster_mean = torch.mean(cluster_embeds, dim=0, keepdim=True)
        
        # 计算与文本嵌入的余弦相似度
        sim = F.cosine_similarity(cluster_mean, text_embeds, dim=1).mean().item()
        part_similarities.append((cluster_id, sim))
    
    # 3. 按相似度排序聚类
    part_similarities.sort(key=lambda x: x[1], reverse=True)
    sorted_clusters = [cid for cid, _ in part_similarities]
    sorted_sims = [sim for _, sim in part_similarities]
    
    # 4. 确定最佳组合（这里使用所有聚类）
    best_clusters = sorted_clusters  # 使用所有聚类
    
    # 5. 生成最终标签（从1开始，未选中的为0）
    final_labels = np.zeros_like(cluster_labels, dtype=int)
    for idx, cluster_id in enumerate(best_clusters):
        final_labels[cluster_labels == cluster_id] = idx + 1  # 标签从1开始
    
    # 转换为PyTorch张量
    final_labels_tensor = torch.from_numpy(final_labels).to(shape_embeds.device)
    
    # 整理返回的信息
    info = {
        "cluster_labels": cluster_labels,
        "similarities": sorted_sims,
        "sorted_clusters": sorted_clusters,
        "best_clusters": best_clusters,
        "kmeans_model": kmeans
    }
    
    return final_labels_tensor, info



class SimilarityAnalyzer:
    """基于相似度变化分析的部件合并工具"""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str, net_type='net1'):
        from model.evaluation.utils import load_model, muti_load_model
        model = load_model(checkpoint_path)
        # model = muti_load_model(checkpoint_path, net_type=net_type)
        model.to(self.device)
        model.eval()
        return model
    
    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> List[float]:
        if s == 0.0:
            return [v, v, v]
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return [v, t, p]
        elif i == 1:
            return [q, v, p]
        elif i == 2:
            return [p, v, t]
        elif i == 3:
            return [p, q, v]
        elif i == 4:
            return [t, p, v]
        else:
            return [v, p, q]
    
    def cluster_parts(self, shape_embeds: torch.Tensor, min_clusters: int = 2, 
                     max_clusters: int = 20, selected_clusters: int = 8) -> torch.Tensor:
        cluster_labels = []
        for num_cluster in range(min_clusters, max_clusters):
            clustering = KMeans(n_clusters=num_cluster, random_state=0).fit(
                shape_embeds.cpu().numpy()
            )
            cluster_labels.append(clustering.labels_)
        
        cluster_labels = np.stack(cluster_labels, axis=0)
        cluster_labels = torch.from_numpy(cluster_labels).to(self.device)
        return cluster_labels[selected_clusters - 2]
    
    def cluster_parts_with_wrapper(self, shape_embeds: torch.Tensor, text_embeds: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """使用封装的函数进行聚类和标签生成"""
        labels, _ = process_embeddings(shape_embeds, text_embeds, n_clusters)
        return labels
    
    def get_part_masks(self, seg_labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        unique_labels = torch.unique(seg_labels)
        part_masks = {}
        
        for label in unique_labels:
            mask = (seg_labels == label)
            part_masks[label] = mask
            
        # 验证掩码有效性
        for label, mask in part_masks.items():
            print(f"部件 {label} 的点数量: {torch.sum(mask).item()}")
                
        print(f"有效分割部件数量：{len(part_masks)}")
        return part_masks
    
    def compute_combined_similarity(self, part_labels: List[int], part_masks: Dict[int, torch.Tensor],
                                   shape_embeds: torch.Tensor, text_embeds: torch.Tensor) -> float:
        """计算组合部件与文本的整体相似度"""
        # 创建组合掩码
        combined_mask = torch.zeros_like(next(iter(part_masks.values())), dtype=torch.bool)
        for label in part_labels:
            combined_mask |= part_masks[label]
        
        # 计算组合特征
        if torch.sum(combined_mask) == 0:
            return 0.0
            
        combined_feat = torch.mean(shape_embeds[combined_mask], dim=0, keepdim=True)
        cos_sim = F.cosine_similarity(combined_feat, text_embeds, dim=1)
        return cos_sim.mean().item()
    
    def sort_parts_by_similarity(self, part_masks: Dict[int, torch.Tensor], 
                                shape_embeds: torch.Tensor, text_embeds: torch.Tensor) -> List[int]:
        part_similarities = []
        for label, mask in part_masks.items():
            part_feat = torch.mean(shape_embeds[mask], dim=0, keepdim=True)
            sim = F.cosine_similarity(part_feat, text_embeds, dim=1).mean().item()
            part_similarities.append((label, sim))
        
        # 按相似度降序排序
        part_similarities.sort(key=lambda x: x[1], reverse=True)
        return [label for label, _ in part_similarities], [sim for _, sim in part_similarities]
    
    def analyze_similarity_changes(self, sorted_labels: List[int], part_masks: Dict[int, torch.Tensor],
                                  shape_embeds: torch.Tensor, text_embeds: torch.Tensor) -> Tuple[List[List[int]], List[float], List[float]]:
        """分析逐步合并过程中的相似度变化"""
        merged_groups = []
        group_similarities = []
        similarity_diffs = []  # 相似度变化率
        current_group = []
        
        for i, label in enumerate(sorted_labels):
            current_group.append(label)
            merged_groups.append(current_group.copy())
            
            # 计算当前组合的整体相似度
            current_sim = self.compute_combined_similarity(current_group, part_masks, shape_embeds, text_embeds)
            group_similarities.append(current_sim)
            
            # 计算相似度变化率
            if i == 0:
                # 第一个组合，没有前序，变化率为0
                similarity_diffs.append(0.0)
            else:
                # 计算与前一个组合的相似度差异
                diff = current_sim - group_similarities[i-1]
                # 计算相对变化率（除以之前的相似度，避免尺度问题）
                relative_diff = diff / abs(group_similarities[i-1]) if group_similarities[i-1] != 0 else 0
                similarity_diffs.append(relative_diff)
            
            print(f"组合 {current_group}: 相似度={current_sim:.6f}, 变化率={similarity_diffs[-1]:.6f}")
        
        return merged_groups, group_similarities, similarity_diffs
    
    def find_optimal_merger(self, merged_groups: List[List[int]], similarities: List[float], diffs: List[float]) -> Tuple[List[int], float]:
        """基于相似度变化找到最佳合并点"""
        if not merged_groups:
            return [], 0.0
            
        # 寻找显著下降点（变化率为负且绝对值较大）
        # 方法1: 寻找第一个明显下降点（变化率小于-0.1）
        for i, diff in enumerate(diffs[1:], 1):  # 从第二个元素开始
            if diff < -0.1:  # 显著下降阈值
                print(f"检测到显著相似度下降在位置 {i}，变化率={diff:.6f}")
                return merged_groups[i-1], similarities[i-1]
        
        # 方法2: 寻找变化率最小的点（最大降幅）
        min_diff_idx = np.argmin(diffs)
        if min_diff_idx > 0:  # 确保不是第一个点
            print(f"最大相似度下降在位置 {min_diff_idx}，变化率={diffs[min_diff_idx]:.6f}")
            return merged_groups[min_diff_idx-1], similarities[min_diff_idx-1]
        
        # 如果没有明显下降，使用整个组合
        return merged_groups[-1], similarities[-1]
    
    def process_object(self, object_path: str, text_embeds: torch.Tensor, 
                      mode: str = 'segmentation', save_path: Optional[str] = None, 
                      use_advanced_clustering: bool = False, n_clusters: int = 8) -> Tuple[List[int], float]:
        # 获取形状嵌入和点云
        from release_tmp.bottle_check.a4_partbasedseman import eval_obj_wild
        shape_embeds, _, shape_pts, _ = eval_obj_wild(
            self.model, object_path, mode, save_path
        )
        
        # 验证获取的特征
        print(f"从模型获取的shape_embeds: 形状={shape_embeds.shape}, 设备={shape_embeds.device}")
        
        # 将数据移至设备
        shape_embeds = shape_embeds.to(self.device)
        shape_pts = shape_pts.to(self.device)
        text_embeds = text_embeds.to(self.device)
        
        # 聚类获取部件（可选择使用新的封装函数）
        if use_advanced_clustering:
            seg_labels = self.cluster_parts_with_wrapper(shape_embeds, text_embeds, n_clusters)
        else:
            seg_labels = self.cluster_parts(shape_embeds)
        print(f"聚类结果: 唯一标签={torch.unique(seg_labels).cpu().numpy()}")
        
        # 获取部件掩码
        part_masks = self.get_part_masks(seg_labels)
        
        # 按相似度排序部件
        sorted_labels, sorted_sims = self.sort_parts_by_similarity(
            part_masks, shape_embeds, text_embeds
        )
        print(f"按相似度排序的部件: {sorted_labels}")
        print(f"对应的相似度: {[f'{s:.6f}' for s in sorted_sims]}")
        
        # 分析合并过程中的相似度变化
        merged_groups, group_sims, sim_diffs = self.analyze_similarity_changes(
            sorted_labels, part_masks, shape_embeds, text_embeds
        )
        
        # 找到最佳合并点
        best_group, best_sim = self.find_optimal_merger(merged_groups, group_sims, sim_diffs)
        print(f"最佳合并组合: {best_group}, 相似度={best_sim:.6f}")
        
        if save_path:
            self.visualize_result(shape_pts, seg_labels, best_group, save_path)
        
        return best_group, best_sim
    
    def visualize_result(self, shape_pts: torch.Tensor, seg_labels: torch.Tensor,
                        best_group: List[int], save_path: str):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shape_pts.cpu().numpy())
        
        target_color = self.hsv_to_rgb(0.6, 0.8, 0.9)  # 蓝色系
        background_color = self.hsv_to_rgb(0, 0, 0.7)  # 灰色系
        
        colors = []
        best_mask = torch.zeros_like(seg_labels, dtype=bool)
        for label in best_group:
            best_mask |= (seg_labels == label)
        
        for is_target in best_mask.cpu().numpy():
            colors.append(target_color if is_target else background_color)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"可视化结果已保存至: {save_path}")
