"""
 单条数据的测试
  cd /apdcephfs_cq11/share_303570626/lanejin/project/find3d_release
  conda activate find3d
  python -m app.segment.eval_benchmark


  除了点云等3D数据，还需要一个mask_labels.txt文件，里面放的是，分割的文本，如
  base
    plant
    body
    neck
    
    mask2points.pt 其实没用到，用空数组代替即可

"""

import os
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel

from app.segment._data import prep_points_val3d

# 必要的外部函数/类依赖（保留核心逻辑）
def get_part_labels(pts_xyz, mask_pts):
    """
    根据mask_pts为点云分配部件标签（修复设备不一致+保持原逻辑）
    Args:
        pts_xyz: 点云张量 [5000, 3]
        mask_pts: 部件掩码张量 [K, 5000]（K为部件数）
    Returns:
        labels: 每个点的部件标签张量 [5000,]（-1表示未匹配，其余为部件索引）
    """
    # 1. 获取输入张量的设备（核心：所有内部张量和输入保持同设备）
    device = pts_xyz.device
    
    # 2. 维度校验（保持原断言逻辑）
    assert pts_xyz.shape == (5000, 3), f"pts_xyz形状应为[5000, 3]，实际为{pts_xyz.shape}"
    K, N = mask_pts.shape
    assert N == 5000, f"mask_pts第二维应为5000，实际为{N}"
    
    # 3. 所有张量创建/转换时显式指定设备（修复核心）
    mask_int = mask_pts.to(torch.int32).to(device)  # 确保在目标设备
    num_activated = mask_int.sum(dim=0)
    
    # labels初始化：指定设备，和输入保持一致
    labels = torch.full((5000,), -1, dtype=torch.long, device=device)
    valid_mask = num_activated == 1  # 自动继承mask_int的设备
    
    if valid_mask.any():
        # 切片后的张量仍保持设备一致
        valid_mask_slice = mask_int[:, valid_mask].to(torch.int32)  # 已在目标设备，to仅确保类型
        valid_indices = torch.argmax(valid_mask_slice, dim=0)  # 自动在目标设备
        labels[valid_mask] = valid_indices  # 设备完全匹配，无报错
    
    return labels

# 核心评价函数（简化版）
def compute_3d_iou_single(net_out, text_embeds, temperature, cat, xyz_sub, xyz_full, gt_full, 
                        save_path, N_CHUNKS=1):
    # -------------------------- 1. 计算预测标签（原有逻辑保留） --------------------------
    # 计算文本-点云相似度得分
    logits = net_out @ text_embeds.t() * temperature
    # 预测标签（从1开始，对应不同颜色类别）
    pred_labels = torch.argmax(logits, dim=1) + 1  
    
    # -------------------------- 2. 上采样预测标签到完整点云（原有逻辑保留） --------------------------
    xyz_full = xyz_full.squeeze()  # 去除冗余维度
    chunk_len = xyz_full.shape[0] // N_CHUNKS + 1
    closest_idx_list = []
    
    # 分块计算最近邻，避免显存溢出
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)].cuda()
        # 计算当前块与下采样点云的距离
        dist = torch.norm(xyz_sub.unsqueeze(0) - cur_chunk.unsqueeze(1), dim=-1)
        # 找到最近邻索引
        min_idxs = torch.min(dist, 1)[1]
        closest_idx_list.append(min_idxs)
    
    # 合并所有块的索引并映射到完整点云的预测标签
    all_nn_idxs = torch.cat(closest_idx_list, axis=0)
    pred_full = pred_labels[all_nn_idxs].cpu().numpy()  # [N_full,] 转为numpy数组
    
    # -------------------------- 3. 定义固定颜色映射表 --------------------------
    # 可根据类别数量扩展/调整颜色，键为预测标签值，值为(R, G, B) 0-255
    color_map = {
        1: (255, 0, 0),      # 红色
        2: (255, 255, 0),    # 黄色
        3: (0, 0, 255),      # 蓝色
        4: (0, 255, 0),      # 绿色
        5: (0, 255, 255),    # 青色
        6: (255, 0, 255),    # 品红/紫色
        7: (128, 128, 128),  # 灰色
        8: (255, 165, 0),    # 橙色
        9: (139, 69, 19),    # 棕色
        10: (240, 230, 140)  # 卡其色
    }
    # 默认颜色（未匹配到的标签使用黑色）
    default_color = (0, 0, 0)
    
    # -------------------------- 4. 处理点云坐标和颜色 --------------------------
    # 转换完整点云坐标为numpy数组
    xyz_np = xyz_full.cpu().numpy()  # [N_full, 3]
    num_points = xyz_np.shape[0]
    
    # 为每个点分配颜色
    colors = []
    for label in pred_full:
        colors.append(color_map.get(int(label), default_color))
    colors = np.array(colors, dtype=np.uint8)  # [N_full, 3]
    
    # -------------------------- 5. 保存为PLY文件 --------------------------
    # 写入PLY头部（ASCII格式）
    with open(save_path, 'w') as f:
        # PLY文件头部定义
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {num_points}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        # 逐行写入点坐标和颜色
        for i in range(num_points):
            x, y, z = xyz_np[i]
            r, g, b = colors[i]
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
    
    print(f"预测结果已保存为PLY文件：{save_path}")
    return save_path

# 单条数据加载函数（复用Eval3dcom核心逻辑）
def load_single_data(data_path, category, textembeds="clip", decorated=True):
    """加载单条3DCompat数据（修复设备不一致问题）"""
    # 1. 统一设备（优先CUDA，无则CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return_dict = {}
    
    # 2. 加载核心数据（移除.cpu()，统一移到目标设备）
    mask_pts = torch.load(f"/x2robot_v2/lanejin/new_data/dataset/mask2points.pt", map_location=device)
    pts_xyz = torch.load(f"{data_path}/Scan_points_can.pt", map_location=device)
    normal = torch.load(f"{data_path}/Scan_normals.pt", map_location=device)
    pts_rgb = torch.load(f"{data_path}/Scan_rgb.pt", map_location=device) * 255
    # # debug 去掉颜色
    # pts_rgb = torch.zeros_like(pts_rgb)
    gt = get_part_labels(pts_xyz, mask_pts) + 1  # 0留给unknown
    gt = gt.to(device)  # 确保gt在目标设备
    
    # 3. 文本编码（确保文本模型和输入都在目标设备）
    with open(f"/x2robot_v2/lanejin/new_data/dataset/omniobject3d_semanticname/{category}_mask_labels.txt", "r") as f:
        labels = f.read().splitlines()
    if decorated:
        labels = [f"{part} of a {category}" for part in labels]
    
    # 初始化文本模型（移到目标设备）
    if textembeds == 'mpnet':
        model_name = "sentence-transformers/all-mpnet-base-v2"
    else:
        model_name = "google/siglip-base-patch16-224"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModel.from_pretrained(model_name).eval().to(device)  # 模型移到设备
    
    with torch.no_grad():
        # 文本输入张量移到目标设备
        inputs = tokenizer(
            labels, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).to(device)  # 关键：输入移到设备
        
        if textembeds == 'mpnet':
            outputs = text_model(**inputs)
            token_embeddings = outputs[0]
            input_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
            text_feat = torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)
        else:
            text_feat = text_model.get_text_features(**inputs)
    
    # 归一化并确保在目标设备
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    text_feat = text_feat.to(device)

    # 4. 数据增强（确保返回的张量都在目标设备）
    return_dict = prep_points_val3d(pts_xyz.cpu().numpy(), pts_rgb.cpu().numpy(), normal.cpu().numpy(), gt.cpu().numpy(), pts_xyz.cpu().numpy(), gt.cpu().numpy())

    # 全部放到device上
    for k in return_dict:
        if isinstance(return_dict[k], torch.Tensor):
            return_dict[k] = return_dict[k].to(device)
    
    # 5. 填充返回字典（所有张量统一设备）
    return_dict['label_embeds'] = text_feat.to(device)
    return_dict['class_name'] = category
    return_dict["xyz_visualization"] = pts_xyz.float().to(device)  # 直接复用已在设备的pts_xyz
    return_dict["offset"] = return_dict["offset"].to(device)  # 确保offset在设备
    
    # 可选：如果需要显式指定cuda（仅当确定有GPU时）
    # for k, v in return_dict.items():
    #     if isinstance(v, torch.Tensor):
    #         return_dict[k] = v.cuda()
    
    return return_dict

# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="单条3DCompat数据评价")
    # ========== 核心修改：添加默认值，移除required=True ==========
    parser.add_argument("--data_path", 
                        type=str, 
                        default="/apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/data/coarse_b29_0cb", 
                        help="单条数据的路径（如xxx/coarse_0001）")
    parser.add_argument("--category", 
                        type=str, 
                        default="vast", 
                        help="数据类别（如chair、table）")
    parser.add_argument("--checkpoint_path", 
                        type=str, 
                        default="dataset/checkpoints/ours_final.pth", 
                        help="模型权重路径")
    parser.add_argument("--save_path", 
                        type=str, 
                        default="/apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/results/org.ply",)
    parser.add_argument("--net_type", 
                        type=str, 
                        default="net8", 
                        help="网络类型（net1/net8等）")
    parser.add_argument("--textembeds", 
                        type=str, 
                        default="clip", 
                        help="文本嵌入类型（clip/mpnet）")
    args = parser.parse_args()

    args.data_path = "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/belt_001"
    args.save_path = "/x2robot_v2/lanejin/new_data/cosmo3d/results_eval2vis/realworld/sampling.ply"
    # args.checkpoint_path = "/x2robot_v2/lanejin/new_data/cosmo3d/dataset/checkpoints/0104_color_ckpt_200.pth"
    args.checkpoint_path = "dataset/checkpoints/ours_final.pth"
    args.category = "belt"

    # 1. 加载模型
    torch.manual_seed(123)
    if args.net_type in ['net1', 'net2']:
        from model.evaluation.utils import load_model
        model = load_model(args.checkpoint_path)
    else:
        from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
        model.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"], strict=True)
    
    model = model.eval().cuda()
    temperature = np.exp(model.ln_logit_scale.item()) if hasattr(model, 'ln_logit_scale') else 1.0

    # 2. 加载单条数据
    data = load_single_data(args.data_path, args.category, args.textembeds)

    # 3. 模型推理
    with torch.no_grad():
        data['mask_offset'] = torch.tensor([data['label_embeds'].shape[0]], device="cuda")
        model_output = model(data)
        
        # 提取模型输出
        if isinstance(model_output, (tuple, list)):
            net_out = model_output[0]
        elif isinstance(model_output, torch.Tensor):
            net_out = model_output
        else:
            raise TypeError(f"不支持的模型输出格式：{type(model_output)}")

    # 4. 计算3D IoU
    save_path = compute_3d_iou_single(
        net_out=net_out,
        text_embeds=data['label_embeds'],
        temperature=temperature,
        cat=data['class_name'],
        xyz_sub=data['coord'],
        xyz_full=data['xyz_full'],
        gt_full=data['gt_full'],
        save_path=args.save_path
    )

    # 5. 输出结果
    print(f"单条数据评价结果：")
    print(f"数据路径: {args.data_path}")
    print(f"类别: {args.category}")
    print(f"保存路径: {save_path}")
