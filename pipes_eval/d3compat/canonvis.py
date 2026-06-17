import os
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
from model.evaluation.core import visualize_pt_labels
from common.utils import visualize_pt_heatmap

import time
from model.evaluation.utils import set_seed, load_model
from transformers import AutoTokenizer, AutoModel

from pipes_eval.d3compat.data import Eval3dcom, collate_fn
from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
# ---------------------- 新增导入：PCA相关 ----------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def save_colored_point_cloud(xyz, colors, save_path):
    """保存带RGB颜色的点云为PLY文件（用于Canonical Color和BBox可视化）"""
    assert xyz.shape[0] == colors.shape[0], "点云坐标与颜色数量不匹配"
    assert colors.shape[1] == 3, "颜色需为RGB三通道格式"

    # 颜色归一化到0-255整数范围
    colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
    colors = (colors * 255).astype(np.uint8)

    # 写入PLY文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {xyz.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for x, y, z, r, g, b in zip(xyz[:, 0], xyz[:, 1], xyz[:, 2], colors[:, 0], colors[:, 1], colors[:, 2]):
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')

# ---------------------- 新增函数：特征PCA降维转颜色 ----------------------
def feat2color_via_pca(feat, random_state=123):
    """
    将高维特征通过PCA降维到3维，转换为可用于点云可视化的颜色数据
    :param feat: 高维特征数组 (N, D)，N为点云数量，D为特征维度
    :param random_state: PCA随机种子，保证结果可复现
    :return: 3维颜色数据 (N, 3)
    """
    # 1. 输入校验
    if len(feat.shape) != 2:
        raise ValueError(f"特征需为2维数组 (N, D)，当前输入形状为 {feat.shape}")
    N, D = feat.shape
    if D < 3:
        # 若特征维度小于3，补零到3维（保证RGB格式）
        color_3d = np.zeros((N, 3), dtype=np.float32)
        color_3d[:, :D] = feat
        return color_3d
    if N <= 1:
        raise ValueError(f"点云数量需大于1，当前输入数量为 {N}")
    
    # 2. 特征标准化（PCA前必须做，消除量纲影响）
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)
    
    # 3. PCA降维到3维
    pca = PCA(n_components=3, random_state=random_state)
    feat_pca = pca.fit_transform(feat_scaled)
    
    # 4. 返回降维结果（后续在save_colored_point_cloud中做0-255归一化）
    return feat_pca

def save_3d_visualization_results(model_outputs, data, xyz_sub, xyz_full, gt_full, 
                                  N_CHUNKS=1, xyz_visualization=None, save_base_path=None):
    """
    保存分割结果、Canonical Color和BBox可视化结果（新增：PCA降维特征可视化）
    :param model_outputs: 模型完整输出 (backbone_feat, decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset)
    :param save_base_path: 保存基础路径（不含后缀）
    """
    # 解析模型输出
    backbone_feat, decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset = model_outputs

    device = canoncolor_out.device
    sigmoid = torch.nn.Sigmoid()

    # ---------------------- 2. 处理并保存Canonical Color ----------------------
    if save_base_path is not None and canoncolor_out is not None:
        # 上采样Canonical Color到完整点云
        # 修复：all_nn_idxs 未定义问题，先从data中尝试获取（兼容原有逻辑）
        all_nn_idxs = data.get('all_nn_idxs', torch.tensor([]))
        if 'all_nn_idxs' in locals() and all_nn_idxs.numel() > 0:
            canoncolor_full = canoncolor_out[all_nn_idxs].cpu().numpy()
        else:
            canoncolor_full = canoncolor_out.cpu().numpy()
        
        xyz_sub_np = xyz_sub.cpu().numpy()
        
        # 保存Canonical Color可视化
        canon_color_path = f"{save_base_path}_canoncolor.ply"
        save_colored_point_cloud(xyz_sub_np, canoncolor_full, canon_color_path)
    
    # ---------------------- 新增：处理并保存PCA降维特征可视化 ----------------------
    if save_base_path is not None and backbone_feat is not None:
        # 1. 提取并转换backbone特征（从GPU转到CPU，张量转numpy）
        # 兼容可能的批次维度，挤压多余维度
        backbone_feat_np = backbone_feat.detach().cpu().numpy()
        backbone_feat_np = np.squeeze(backbone_feat_np)  # 去除形状为1的批次维度/通道维度
        
        # 2. 确保点云坐标和特征数量匹配
        xyz_sub_np = xyz_sub.cpu().numpy()
        if backbone_feat_np.shape[0] != xyz_sub_np.shape[0]:
            print(f"警告：特征数量 {backbone_feat_np.shape[0]} 与点云数量 {xyz_sub_np.shape[0]} 不匹配，跳过PCA可视化")
        else:
            # 3. PCA降维转颜色
            feat_color = feat2color_via_pca(backbone_feat_np)
            
            # 4. 保存PCA特征可视化结果（文件名带pca_feat标识，方便区分）
            pca_feat_path = f"{save_base_path}_pca_feat.ply"
            save_colored_point_cloud(xyz_sub_np, feat_color, pca_feat_path)
            print(f"PCA特征可视化结果已保存至：{pca_feat_path}")

def process_3d_dataset(model, dataloader, N_CHUNKS=1, save_dir=None, testname=None):
    """批量处理数据集，保存所有可视化结果"""
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # 数据设备迁移
            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and "full" not in key:
                    data[key] = data[key].cuda(non_blocking=True)

            # 构建mask_offset并获取模型输出
            data['mask_offset'] = torch.tensor(
                [data['label_embeds'].shape[0]],
                device=data['offset'].device
            )
            model_outputs = model(data)

            # 提取基础数据
            gt_full = data["gt_full"]
            xyz_sub = data["coord"]
            xyz_full = data["xyz_full"]
            cat = data["class_name"][0]
            uidname = data.get("uidname", [str(idx)])[0]
            xyz_visualization = data["xyz_visualization"]

            # 设置保存路径（限制每个类别最多保存10个样本）
            save_base_path = None
            if save_dir is not None:
                cat_save_dir = os.path.join(save_dir, cat)
                os.makedirs(cat_save_dir, exist_ok=True)
                
                # 统计已保存的PLY文件数量（去重，每个样本只计数一次）
                existing_samples = set()
                for f in os.listdir(cat_save_dir):
                    if f.endswith(".ply"):
                        sample_name = f.split("_")[0]
                        existing_samples.add(sample_name)
                
                if len(existing_samples) < 10:
                    save_base_path = os.path.join(cat_save_dir, uidname)

            
            # pre模式保存所有结果
            save_3d_visualization_results(model_outputs, data, xyz_sub, xyz_full, gt_full,
                                              N_CHUNKS, xyz_visualization, save_base_path)

def eval_category(data_root, category, model, apply_rotation=False, subset=False,
                  decorated=True, save_dir=None, textembeds=None, datatype=None, testname=None):
    """处理单个类别数据"""
    # 构建数据集和数据加载器
    test_data = Eval3dcom(data_root, category, textembeds=textembeds, datatype=datatype,
                          apply_rotation=apply_rotation, decorated=decorated)
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False
    )

    # 处理数据并计时
    start_time = time.time()
    process_3d_dataset(model, test_loader, N_CHUNKS=20, save_dir=save_dir, testname=testname)
    elapsed_time = time.time() - start_time

    print(f"类别 {category} 处理完成，耗时: {elapsed_time:.2f} 秒")
    return category, elapsed_time

def batch_eval(data_root, model, apply_rotation, subset, decorated, save_path=None,
               textembeds=None, datatype=None, testname=None):
    """批量处理所有类别"""
    # 获取所有类别
    category_names = os.listdir(data_root)
    print(f"总共有 {len(category_names)} 个类别待处理")

    # 初始化保存目录
    total_time = 0.0
    save_dir = os.path.dirname(save_path) if save_path is not None else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 处理每个类别
    category_times = []
    for cat in category_names:
        print("-" * 50)
        print(f"开始处理类别: {cat}")
        category, cost_time = eval_category(
            data_root, cat, model,
            apply_rotation=apply_rotation,
            subset=subset,
            decorated=decorated,
            save_dir=save_dir,
            textembeds=textembeds,
            datatype=datatype,
            testname=testname
        )
        category_times.append((category, cost_time))
        total_time += cost_time

    # 打印统计结果
    print("-" * 50)
    print("所有类别处理完成，统计信息如下:")
    print(f"{'类别名称':<20} {'耗时(秒)':<10}")
    print("-" * 30)
    for cat, t in category_times:
        print(f"{cat:<20} {t:<10.2f}")
    print("-" * 30)
    print(f"总耗时: {total_time:.2f} 秒")

    # 保存统计结果到文本文件
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"{'类别名称':<20} {'耗时(秒)':<10}\n")
            f.write("-" * 30 + "\n")
            for cat, t in category_times:
                f.write(f"{cat:<20} {t:<10.2f}\n")
            f.write("-" * 30 + "\n")
            f.write(f"总耗时: {total_time:.2f} 秒\n")

if __name__ == '__main__':
    # 设置随机种子
    set_seed(123)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="3D点云分割、Canonical Color和BBox可视化结果保存（新增PCA特征降维可视化）")
    parser.add_argument("--benchmark", required=True, type=str, help="基准数据集: Objaverse, ShapeNetPart, PartNetE")
    parser.add_argument("--data_root", required=True, type=str, help="数据集根目录")
    parser.add_argument("--save_dir", required=True, type=str, help="结果保存根目录")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="模型权重文件路径")
    parser.add_argument("--d3com_datatype", required=True, type=str, help="d3compat数据类型: coarse/fine")
    parser.add_argument("--net_type", required=True, type=str, help="网络架构类型: net1/net2/.../net8")
    parser.add_argument("--test_type", required=True, type=str, help="测试方式: feats/pre")

    parser.add_argument("--objaverse_split", type=str, help="Objaverse数据集分割: seenclass/unseen/shapenetpart")
    parser.add_argument("--canonical", action='store_false', dest="rotate", help="是否关闭随机旋转")
    parser.add_argument("--subset", action='store_true', dest="subsample", help="是否仅评估子集")
    parser.add_argument("--part_query", action='store_false', dest="decorate", help="是否使用简化查询提示")
    parser.add_argument("--use_shapenetpart_topk_prompt", action='store_true', help="是否使用TopK查询提示")
    parser.add_argument("--textembeds", type=str, default="clip", help="文本嵌入类型 (默认: clip)")

    parser.set_defaults(rotate=True, subsample=False, decorate=True, use_shapenetpart_topk_prompt=False)
    args = parser.parse_args()

    # 加载模型
    if args.net_type in ['net1', 'net2']:
        model = load_model(args.checkpoint_path)
    elif args.net_type == 'net3':
        from release_pipeline3.stage1_semanspace.halfd3com_worot_aligncates_decoder.mixdecodernet import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == 'net4':
        from release_pipeline5ab.ab4_partfieldloss_sizeaug_decoder_canoncolor.mixdecodernet import PointSemSegWithDecoder_test as PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == 'net5':
        from release_pipeline5ab.ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign.mixdecodernet import PointSemSegWithDecoder_test as PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == 'net6':
        from release_pipeline5ab.ab6_partfieldloss_sizeaug_decoder_bbox.mixdecodernet import PointSemSegWithbboxDecoder
        model = PointSemSegWithbboxDecoder(args=args)
    elif args.net_type == 'net7':
        from release_pipeline6.ab1_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox.mixdecodernet import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == 'net8':
        from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    else:
        raise ValueError(f"不支持的网络类型: {args.net_type}")

    # 加载模型权重并设置为评估模式
    model.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"], strict=True)
    model.eval()
    model = model.cuda()

    # 处理d3compat数据集
    if args.benchmark == "d3compat":
        # 构建保存路径
        traindataname = args.checkpoint_path.split('/')[1]
        save_filename = f"d3compat_{traindataname}_{args.net_type}_{args.d3com_datatype}_{args.test_type}"
        if args.decorate:
            save_filename += "_partcates"
        if args.rotate:
            save_filename += "_roted"
        save_filename += ".txt"
        save_path = os.path.join(args.save_dir, save_filename)

        # 执行批量评估
        batch_eval(
            args.data_root, model,
            apply_rotation=args.rotate,
            subset=args.subsample,
            decorated=args.decorate,
            save_path=save_path,
            textembeds=args.textembeds,
            datatype=args.d3com_datatype,
            testname=args.test_type
        )