import os
# 添加解释器路径
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


import torch
import numpy as np
from model.evaluation.core import visualize_3d_upsample
import numpy as np
import argparse
from model.evaluation.utils import set_seed, load_model, read_pcd, encode_text  # , preprocess_pcd
from release_module.obj2ptscolornorm.obj2ptscolornorm import load_and_sample_mesh


from model.data.augmentation import *
def prep_points_train(xyz, rgb, normal):
    # xyz, rgb, normal 均为 (n,3) 的 numpy 数组
    # rgb 取值范围为 0-255
    # 首先转换坐标系：x 轴反转，y 和 z 轴互换
    xyz_change_axis = np.concatenate([
        -xyz[:, 0].reshape(-1, 1),  # x 轴反转
        xyz[:, 2].reshape(-1, 1),   # 原 z 轴作为新 y 轴
        xyz[:, 1].reshape(-1, 1)    # 原 y 轴作为新 z 轴
    ], axis=1)
    
    # 构建数据字典（移除 mask2pt）
    data_dict = {
        "coord": xyz_change_axis, 
        "color": rgb, 
        "normal": normal
    }
    
    # 数据增强和处理（移除与 mask2pt 相关的操作）
    data_dict = CenterShift(apply_z=True)(data_dict)  # 中心偏移（含 z 轴）
    data_dict = RandomScale(scale=[0.9, 1.1])(data_dict)  # 随机缩放
    data_dict = RandomFlip(p=0.5)(data_dict)  # 随机翻转
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)  # 随机抖动
    data_dict = ChromaticAutoContrast(p=0.2, blend_factor=None)(data_dict)  # 颜色自动对比度
    data_dict = ChromaticTranslation(p=0.95, ratio=0.05)(data_dict)  # 颜色平移
    data_dict = ChromaticJitter(p=0.95, std=0.05)(data_dict)  # 颜色抖动
    data_dict = GridSample(
        grid_size=0.02, 
        hash_type='fnv', 
        mode='train', 
        return_grid_coord=True
    )(data_dict)  # 网格采样
    data_dict = CenterShift(apply_z=False)(data_dict)  # 中心偏移（不含 z 轴）
    data_dict = NormalizeColor()(data_dict)  # 颜色归一化
    data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)  # 添加条件标签
    
    # 转换为张量并收集关键数据（移除 mask2pt 相关的 offset）
    data_dict = ToTensor()(data_dict)  # 转换为 PyTorch 张量
    data_dict = Collect(
        keys=('coord', 'grid_coord'),  # 保留坐标和网格坐标
        offset_keys_dict={"offset": "coord"},  # 仅保留坐标的 offset
        feat_keys=('color', 'normal')  # 特征保留颜色和法向量
    )(data_dict)
    
    return data_dict

# data is a dict that has gone through preprocessing as training (normalizing etc.)
def visualize_seg3d(model, data, mode, N_CHUNKS=5, savepath=None): # evaluate loader can only have batch size=1
    if mode == "segmentation":
        heatmap = False
    elif mode == "heatmap":
        heatmap = True
    else:
        print("unsupported mode")
        return
    start_time = time.time()
    temperature = np.exp(model.ln_logit_scale.item())
    with torch.no_grad():
        for key in data.keys():
            if isinstance(data[key], torch.Tensor) and "full" not in key:
                data[key] = data[key].cuda(non_blocking=True)
        net_out = model(x=data)  # 这个里面只有对点云的处理 ； 无label_embeds

        '''# debug
        for key, value in data.items():
            print(key, value.shape)
        print('net_out:', net_out.shape)
        asdf'''

        print(f"-----------------Model inference done in {time.time() - start_time:.2f} seconds")
        text_embeds = data['label_embeds']
        xyz_sub = data["coord"]
        # xyz_full = data["xyz_full"]
        xyz_full = data["coord"]
        caption_list = visualize_3d_upsample(net_out, # n_subsampled_pts, feat_dim
                            text_embeds, # n_parts, feat_dim
                            temperature,
                            xyz_sub,
                            xyz_full, # n_pts, 3
                            panoptic=False,
                            N_CHUNKS=N_CHUNKS,
                            heatmap=heatmap,
                            savepath=savepath)
    return caption_list


'''def eval_obj_wild(model, obj_path, mode, queries, savepath):
    if mode not in ["segmentation", "heatmap"]:
        print("only segmentation or heatmap mode are supported")
        return
    # 如果是pcd文件，读取xyz, rgb, normal
    if obj_path.endswith('.pcd'):
        xyz, rgb, normal = read_pcd(obj_path,visualize=False)  # (torch.Size([5000, 3]), torch.Size([5000, 3]), torch.Size([5000, 3]))
    elif obj_path.endswith('.obj'):
        xyz, rgb, normal = load_and_sample_mesh(obj_path, 5000, colornormflag=True)  # (obj_path, visualize=False, num_points=5000, return_normal=True)  # (torch.Size([5000, 3]), torch.Size([5000, 3]), torch.Size([5000, 3]))

    start_time = time.time()
    data_dict = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
    data_dict["label_embeds"] = encode_text(queries)
    print(f"-----------------Data preprocessed in {time.time() - start_time:.2f} seconds")
    visualize_seg3d(model, data_dict, mode, savepath=savepath)
    return'''


# 验证合并的是不是对
def save_partial_pointclouds(xyz, merged_mask_pts, merged_queries, output_dir="output_pointclouds"):
    """
    从xyz点云中根据merged_mask_pts提取对应的部分，并保存为PLY文件
    
    参数:
        xyz: 完整的点云坐标 tensor
        merged_mask_pts: 合并后的mask tensor
        merged_queries: 合并后的查询标签列表
        output_dir: 输出PLY文件的目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保mask和queries数量匹配
    assert len(merged_mask_pts) == len(merged_queries), \
        "merged_mask_pts和merged_queries的数量必须一致"
    
    # 遍历每个合并后的部分
    for i, (mask, query) in enumerate(zip(merged_mask_pts, merged_queries)):
        # 将mask转换为布尔值（假设mask是0/1表示）
        # 如果mask是概率值，可能需要先进行阈值处理，例如 mask > 0.5
        mask_bool = mask.bool()
        
        # 根据mask从xyz中提取对应的点
        partial_xyz = xyz[mask_bool]
        
        # 如果没有提取到点，跳过
        if len(partial_xyz) == 0:
            print(f"警告: 对于查询 '{query}' 没有提取到任何点，跳过保存")
            continue
        
        # 处理文件名，移除可能的无效字符
        filename = f"{query.replace(' ', '_').replace('/', '_')}.ply"
        file_path = os.path.join(output_dir, filename)
        
        # 保存为PLY文件
        with open(file_path, 'w') as f:
            # PLY文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(partial_xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # 写入点坐标
            if isinstance(partial_xyz, torch.Tensor):
                partial_xyz_np = partial_xyz.numpy()
            else:
                partial_xyz_np = partial_xyz
                
            for point in partial_xyz_np:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        print(f"已保存点云部分: {file_path}，包含 {len(partial_xyz)} 个点")

# 测试数据集
def eval_obj_wild(model, data_dir, mode, savepath):
    if mode not in ["segmentation", "heatmap"]:
        print("only segmentation or heatmap mode are supported")
        return
    '''# 如果是pcd文件，读取xyz, rgb, normal
    if obj_path.endswith('.pcd'):
        xyz, rgb, normal = read_pcd(obj_path,visualize=False)  # (torch.Size([5000, 3]), torch.Size([5000, 3]), torch.Size([5000, 3]))
    elif obj_path.endswith('.obj'):
        xyz, rgb, normal = load_and_sample_mesh(obj_path, 5000, colornormflag=True)  # (obj_path, visualize=False, num_points=5000, return_normal=True)  # (torch.Size([5000, 3]), torch.Size([5000, 3]), torch.Size([5000, 3]))'''
    
    # 判断obj_path是否为文件夹
    if os.path.isdir(data_dir):
        with open(f"{data_dir}/mask_labels.txt", "r") as f:
            queries = f.read().splitlines()
        mask_pts = torch.load(f"{data_dir}/mask2points.pt").cpu()
        xyz = torch.load(f"{data_dir}/points.pt").cpu()
        normal = torch.load(f"{data_dir}/normals.pt").cpu()
        rgb = torch.load(f"{data_dir}/rgb.pt").cpu() # *255

        # debug ； 将重复的标签进行合并
        queries_np = np.array(queries)  # 1. 建立标签到索引的映射
        unique_labels, label_ids = np.unique(queries_np, return_inverse=True)
        label_ids = torch.from_numpy(label_ids).to(mask_pts.device)  # 确保在同一设备
        n_unique = len(unique_labels)  # 2. 合并mask_pts（兼容旧版PyTorch的实现）
        merged_mask_pts = torch.zeros(n_unique, mask_pts.size(1), 
                                    dtype=mask_pts.dtype, 
                                    device=mask_pts.device)
        for i in range(n_unique):  # 对每个唯一标签执行聚合操作
            mask = (label_ids == i)  # 找到属于当前标签的所有行索引
            if mask.any():
                merged_mask_pts[i] = mask_pts[mask].max(dim=0).values  # 取这些行的最大值（实现OR操作）
        merged_queries = unique_labels.tolist()  # 3. 整理结果
        
        # 替换原本的标签
        queries = merged_queries
        mask_pts = merged_mask_pts

        ''''''# 验证合并是否正确
        savedir = os.path.join(savepath, "partial_pointclouds",os.path.basename(data_dir))
        save_partial_pointclouds(xyz, mask_pts, queries, savedir)
        print('savedir:', savedir)



    start_time = time.time()
    # data_dict = preprocess_pcd(xyz.cuda(), rgb.cuda(), normal.cuda())
    data_dict = prep_points_train(xyz.numpy(), rgb.numpy(), normal.numpy())
    data_dict["label_embeds"] = encode_text(queries)
    print(f"-----------------Data preprocessed in {time.time() - start_time:.2f} seconds")
    caption_list = visualize_seg3d(model, data_dict, mode, savepath=savepath)
    
    
    # 打印queries 和 
    print(f'queries: {queries}')
    quaryColor = {}
    for i in range(len(caption_list)):
        if i == 0 : 
            continue
        quaryColor[queries[i-1]] =  caption_list[i].split(':')[-1]
    print('quaryColor : ', quaryColor)

    return


if __name__ == '__main__':
    set_seed(123)
    parser = argparse.ArgumentParser(description="Please specify input point cloud path and model checkpoint path")
    parser.add_argument("--object_path", required=True, type=str, help='The point cloud to evaluate on. Should be a .pcd file')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='path of the checkpoint to evaluate')
    parser.add_argument("--mode", required=True, type=str, help='segmentation or heatmap')
    # parser.add_argument("--queries", required=True, nargs='+', help='list of queries')
    parser.add_argument("--savepath", type=str, default=None, help='保存分割结果的路径')
    args = parser.parse_args()
    

    start_time = time.time()
    model = load_model(args.checkpoint_path)
    print(f"-----------------Model loaded in {time.time() - start_time:.2f} seconds")
    # eval_obj_wild(model,args.object_path, args.mode, args.queries, args.savepath)
    eval_obj_wild(model,args.object_path, args.mode, args.savepath)

    