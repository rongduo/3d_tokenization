"""
 单条数据的测试
  cd /apdcephfs_cq11/share_303570626/lanejin/project/find3d_release
  conda activate find3d
  python -m app.segment.eval_benchmark_batch


  除了点云等3D数据，还需要一个mask_labels.txt文件，里面放的是，分割的文本，如
  base
    plant
    body
    neck
    
    mask2points.pt 其实没用到，用空数组代替即可

"""
import argparse
import torch
import numpy as np
import os
from pathlib import Path
from  tqdm import tqdm

# 注意：保留原有代码中的导入（根据你的项目环境，确保以下导入可用）
# 这里保留原有核心导入，若缺少可补充
from model.evaluation.utils import load_model
from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
from app.segment.eval_benchmark import load_single_data  # 替换为你实际的导入路径

# 定义颜色（RGB格式，取值0-255）
TARGET_COLOR = (255, 165, 0)  # 橙色（目标语义部分，如chair的leg）
OTHER_COLOR = (245, 245, 220)  # 米色（其他非目标语义部分）

# 核心评价函数（修改版：新增seg.txt和semantic.txt保存）
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
    
    # -------------------------- 5. 保存为PLY文件（原有逻辑保留） --------------------------
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
    
    # -------------------------- 6. 新增：保存seg.txt（每行一个点的标签索引） --------------------------
    # 提取save_path的目录和文件名前缀，用于生成seg.txt的路径
    import os
    save_dir = os.path.dirname(save_path)
    seg_txt_path = os.path.join(save_dir, "seg.txt")
    
    with open(seg_txt_path, 'w', encoding='utf-8') as f:
        for label in pred_full:
            f.write(f"{int(label-1)}\n")  # 转为int确保格式整洁，每行一个标签
    
    # -------------------------- 7. 新增：保存语义对应txt（label → 语义/颜色说明） --------------------------
    semantic_txt_path = os.path.join(save_dir, "semantic.txt")
    # 先获取本次预测中出现的唯一标签（去重并排序）
    unique_labels = sorted(list(np.unique(pred_full)))
    
    with open(semantic_txt_path, 'w', encoding='utf-8') as f:
        # 写入文件头说明
        f.write("标签索引 → 对应颜色 → 语义说明（可根据项目需求补充具体语义名称）\n")
        f.write("=" * 80 + "\n")
        
        # 遍历每个出现的标签，写入对应信息
        for label in unique_labels:
            color = color_map.get(int(label), default_color)
            color_desc = f"R:{color[0]}, G:{color[1]}, B:{color[2]}"
            # 若有具体语义映射（如 1→"chair_leg"），可在此处扩展，示例：
            # semantic_desc = SEMANTIC_MAP.get(int(label), "未知语义")
            semantic_desc = f"语义_{int(label)}"
            f.write(f"{int(label)} → {color_desc} → {semantic_desc}\n")
    
    # -------------------------- 8. 打印保存完成信息 --------------------------
    print(f"预测结果已保存为PLY文件：{save_path}")
    print(f"点标签索引已保存为seg.txt：{seg_txt_path}")
    print(f"标签语义对应关系已保存为semantic.txt：{semantic_txt_path}")
    
    return save_path

def read_obj_file(obj_path):
    """
    读取OBJ文件，提取顶点、面等信息
    """
    vertices = []
    faces = []
    with open(obj_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'v':
                # 读取顶点坐标（x, y, z）
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
            elif parts[0] == 'f':
                # 读取面信息（简化处理，保留原始顶点索引）
                face = [p.split('/')[0] for p in parts[1:]]
                faces.append(face)
    return np.array(vertices), faces

def write_colored_ply(ply_save_path, vertices, faces, semantic_mask, target_semantic_label):
    """
    仅写入带颜色的PLY文件（分割结果可视化），移除OBJ/MTL相关逻辑
    """
    with open(ply_save_path, 'w', encoding='utf-8') as ply_f:
        # PLY文件头
        ply_f.write("ply\n")
        ply_f.write("format ascii 1.0\n")
        ply_f.write(f"element vertex {len(vertices)}\n")
        ply_f.write("property float x\n")
        ply_f.write("property float y\n")
        ply_f.write("property float z\n")
        ply_f.write("property uchar red\n")
        ply_f.write("property uchar green\n")
        ply_f.write("property uchar blue\n")
        ply_f.write(f"element face {len(faces)}\n")
        ply_f.write("property list uchar int vertex_indices\n")
        ply_f.write("end_header\n")
        
        # 写入带颜色的顶点
        for i, vert in enumerate(vertices):
            if i < len(semantic_mask) and semantic_mask[i] == target_semantic_label:
                r, g, b = TARGET_COLOR
            else:
                r, g, b = OTHER_COLOR
            ply_f.write(f"{vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f} {r} {g} {b}\n")
        
        # 写入面
        for face in faces:
            face_indices = [int(v)-1 for v in face if int(v)-1 < len(vertices)]
            if face_indices:
                ply_f.write(f"{len(face_indices)} {' '.join(map(str, face_indices))}\n")
    
    return ply_save_path

def write_seg_txt(seg_txt_path, semantic_mask):
    """
    写入每个点的标签索引seg.txt，每行一个数字
    """
    with open(seg_txt_path, 'w', encoding='utf-8') as f:
        for label in semantic_mask:
            f.write(f"{label}\n")
    return seg_txt_path

def write_semantic_txt(semantic_txt_path, unique_semantic_labels):
    """
    写入标签对应的语义txt（当前为标签索引说明，可根据项目扩展具体语义名称映射）
    格式：标签索引 → 对应语义（默认先保留索引，后续可补充映射关系）
    """
    with open(semantic_txt_path, 'w', encoding='utf-8') as f:
        f.write("标签索引 → 语义名称（可根据项目需求补充具体映射）\n")
        f.write("=" * 50 + "\n")
        for label in sorted(unique_semantic_labels):
            # 若有具体语义映射（如 3→chair_leg），可在此处修改，示例：
            # semantic_name = SEMANTIC_MAP.get(label, f"未知语义_{label}")
            # f.write(f"{label} → {semantic_name}\n")
            f.write(f"{label} → 语义_{label}\n")
    return semantic_txt_path

def process_single_semantic(obj_folder, obj_name, semantic_label, semantic_mask, args):
    """
    处理单个语义部分，仅保存核心PLY文件（移除额外文件夹和OBJ/MTL）
    """
    # 1. 提取类别名（如chair）和语义名（如leg）
    cate_name, _ = obj_name.split("_", maxsplit=1)
    semantic_name = str(semantic_label) if isinstance(semantic_label, int) else semantic_label
    print('  处理语义：', semantic_name)
    semantic_name = f"part_{semantic_name}"
    
    # 2. 定位物体的Scan/Scan.obj路径
    scan_obj_path = os.path.join(obj_folder, "Scan", "Scan.obj")
    if not os.path.exists(scan_obj_path):
        print(f"警告：未找到Scan.obj文件 → {scan_obj_path}，跳过该语义处理")
        return None
    
    # 3. 统一创建保存文件夹（不再创建多级语义子文件夹，简化路径）
    core_save_dir = os.path.join(
        args.save_root,
        cate_name,
        obj_name
    )
    os.makedirs(core_save_dir, exist_ok=True)
    
    # 4. 定义三个核心文件的保存路径
    ply_save_name = f"{obj_name}_{semantic_name}_seg_result.ply"
    ply_save_path = os.path.join(core_save_dir, ply_save_name)
    seg_txt_path = os.path.join(core_save_dir, "seg.txt")
    semantic_txt_path = os.path.join(core_save_dir, "semantic_label.txt")
    
    # 5. 读取原始OBJ文件，写入PLY文件（仅执行一次seg.txt和semantic_txt，避免重复覆盖）
    vertices, faces = read_obj_file(scan_obj_path)
    ply_save_path = write_colored_ply(
        ply_save_path,
        vertices,
        faces,
        semantic_mask,
        target_semantic_label=semantic_label
    )
    
    # 确保seg.txt和semantic_txt仅生成一次（基于唯一标签）
    if not os.path.exists(seg_txt_path):
        write_seg_txt(seg_txt_path, semantic_mask)
    unique_labels = np.unique(semantic_mask)
    if not os.path.exists(semantic_txt_path):
        write_semantic_txt(semantic_txt_path, unique_labels)
    
    print(f"  语义{semantic_name}保存完成：")
    print(f"    PLY文件：{ply_save_path}")
    print(f"    点标签文件：{seg_txt_path}")
    print(f"    语义映射文件：{semantic_txt_path}")
    return core_save_dir

def process_single_object(obj_folder, obj_name, args, model, temperature):
    """
    处理单个物体文件夹（提取类别、uid，执行推理和IoU计算，处理语义分割结果并保存核心文件）
    """
    # 1. 提取类别名和uid编号（按下划线分割，取前半部分为类别，后半部分为uid）
    if "_" not in obj_name:
        print(f"警告：文件夹{obj_name}命名不符合'类别_uid'格式，跳过")
        return
    uid = obj_name.split("_")[-1]
    cate_name = obj_name.replace(f"_{uid}", "")
    print(f"\n===== 开始处理：{obj_name}（类别：{cate_name}，UID：{uid}）=====")
    
    # 2. 动态配置当前物体的路径
    current_data_path = obj_folder
    current_save_dir = os.path.join(
        args.save_root,
        cate_name,
        obj_name
    )
    os.makedirs(current_save_dir, exist_ok=True)
    current_save_path = os.path.join(current_save_dir, f"{obj_name}_sampling.ply")
    
    # 3. 加载单条数据（使用提取的类别名）
    data = load_single_data(current_data_path, cate_name, args.textembeds)
    
    # 4. 模型推理（保持原有逻辑不变）
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
    
    # 5. 计算3D IoU并保存结果（获取语义分割掩码）
    save_path = compute_3d_iou_single(
        net_out=net_out,
        text_embeds=data['label_embeds'],
        temperature=temperature,
        cat=data['class_name'],
        xyz_sub=data['coord'],
        xyz_full=data['xyz_full'],
        gt_full=data['gt_full'],
        save_path=current_save_path
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量处理OmniObject3D数据评价（仅保存PLY+seg.txt+semantic_label.txt）")
    # ========== 核心修改：添加默认值，移除required=True ==========
    parser.add_argument("--data_root",  # 改为数据根路径，不再指定单个数据路径
                        type=str, 
                        default="/x2robot_v2/lanejin/new_data/dataset/omniobject3d", 
                        help="OmniObject3D数据根路径")
    parser.add_argument("--checkpoint_path", 
                        type=str, 
                        default="dataset/checkpoints/ours_final.pth", 
                        help="模型权重路径")
    parser.add_argument("--save_root",  # 改为保存根路径，动态生成子路径
                        type=str, 
                        default="/x2robot_v2/lanejin/new_data/cosmo3d/results_eval2vis/realworld",
                        help="结果保存根路径")
    parser.add_argument("--net_type", 
                        type=str, 
                        default="net8", 
                        help="网络类型（net1/net8等）")
    parser.add_argument("--textembeds", 
                        type=str, 
                        default="clip", 
                        help="文本嵌入类型（clip/mpnet）")
    args = parser.parse_args()

    # ========== 批量处理配置：覆盖默认路径（可选，保持与你原有逻辑一致） ==========
    args.data_root = "/x2robot_v2/lanejin/new_data/dataset/omniobject3d"
    args.save_root = "/x2robot_v2/lanejin/new_data/cosmo3d/results_eval2vis/realworld"
    # args.checkpoint_path = "dataset/checkpoints/ours_final.pth"
    args.checkpoint_path = "dataset/checkpoints/ours_wocolor.pth"
    # args.checkpoint_path = "dataset/checkpoints/orgfind3d.pth"
    # args.net_type = "net1"
    # 如需使用GPU权重，可取消注释下面这行
    # args.checkpoint_path = "/x2robot_v2/lanejin/new_data/cosmo3d/dataset/checkpoints/0104_color_ckpt_200.pth"

    # ========== 1. 加载模型（保持原有逻辑不变，仅执行一次，提高效率） ==========
    torch.manual_seed(123)
    model = None
    if args.net_type in ['net1', 'net2']:
        from model.evaluation.utils import load_model
        model = load_model(args.checkpoint_path)
    else:
        from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
        checkpoint = torch.load(args.checkpoint_path, map_location="cuda")  # 加载到GPU
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    
    model = model.eval().cuda()
    temperature = np.exp(model.ln_logit_scale.item()) if hasattr(model, 'ln_logit_scale') else 1.0
    print("模型加载成功！开始批量处理数据...")
    print("-" * 80)

    # ========== 2. 批量遍历OmniObject3D下所有物体文件夹 ==========
    # 验证数据根路径是否存在
    if not os.path.isdir(args.data_root):
        print(f"错误：数据根路径不存在 → {args.data_root}")
        exit(1)
    
    # # 遍历根路径下所有文件夹（仅处理文件夹，排除文件）
    # for entry in tqdm(os.scandir(args.data_root)):
    path_list = [
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/kettle/kettle_001/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/kettle/kettle_006/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/teapot/teapot_001/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/bottle/bottle_038/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/bottle/bottle_061/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/fork/fork_002/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/glasses/glasses_008/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/glasses/glasses_004/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/handbag/handbag_001/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/handbag/handbag_004/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/handbag/handbag_007/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/handbag/handbag_015/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/handbag/handbag_019/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/handbag/handbag_052/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/handbag/handbag_053/Scan",
        "/x2robot_v2/lanejin/new_data/dataset/omniobject3d/vase/vase_003/Scan"
    ]
    for entry in tqdm(path_list):
        # if entry.is_dir():
        #     obj_folder = entry.path  # 物体文件夹完整路径（如/belt_001）
        #     obj_name = entry.name    # 物体文件夹名称（如belt_001）
        
        obj_folder = entry  # 物体文件夹完整路径（如/belt_001）
        obj_name = entry.split("/")[-2]    # 物体文件夹名称（如belt_001）
        # 处理单个物体（调用封装函数）
        process_single_object(obj_folder, obj_name, args, model, temperature)

    # ========== 3. 批量处理完成 ==========
    print("\n" + "=" * 80)
    print("所有物体处理完毕！核心文件已保存至：", args.save_root)