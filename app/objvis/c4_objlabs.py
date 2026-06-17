"""
将点云的标签转移到model上
cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D2/eval2ivs/objaverse
conda activate base
python -m app.objvis.c4_objlabs
"""
import os
import numpy as np
import trimesh
import torch
from collections import defaultdict, Counter
import matplotlib.cm as cm
from tqdm import tqdm
import seaborn as sns


# -------------------------- 全局配置（适配新路径规则：Scan.obj & Scan_points.pt） --------------------------
# 1. seg.txt搜索根目录（realworld，结构：{category}/{instance}/seg.txt）
SEG_TXT_SEARCH_ROOT = "/x2robot_v2/lanejin/new_data/cosmo3d/results_eval2vis/realworld"
# 2. 数据集根目录（omniobject3d，结构：{category}/{instance}/Scan/）
DATASET_ROOT = "/x2robot_v2/lanejin/new_data/dataset/omniobject3d"
# 3. 模型与点云配置（固定文件名）
SCAN_DIR_NAME = "Scan"                # 模型/点云所在子目录名
TARGET_OBJ_NAME = "Scan.obj"          # 目标obj文件名（固定：Scan.obj）
TARGET_POINT_NAME = "Scan_points.pt"  # 目标点云文件名（固定：Scan_points.pt）
# 4. 输出文件夹配置
OUTPUT_FOLDER_NAME = "seg_obj"        # 输出文件夹（uid/instance目录下）
# 5. 加速参数（保持原配置）
FACE_BATCH_SIZE = 1000
MAX_FILL_ITER = 50
# 6. 材质与颜色配置（保留）
MATERIAL_NAME_PREFIX = "mat_"           # 材质名称前缀
OTHERS_LABEL = "others"                 # 非当前语义标签名
OTHERS_COLOR = np.array([128, 128, 128], dtype=np.uint8)  # others默认灰色（0-255）


# -------------------------- 核心工具函数（重构路径推导逻辑） --------------------------
def load_batch_data(seg_dir_path):
    """
    加载单模型的「点云、模型、标签」，适配新路径规则：
    1. 从seg_dir_path解析category和instance
    2. 推导数据集路径：DATASET_ROOT/{category}/{instance}/Scan/
    3. 加载Scan.obj和Scan_points.pt
    """
    seg_txt_path = os.path.join(seg_dir_path, "seg.txt")
    
    # 解析category和instance（从seg_dir_path提取：seg_root/category/instance → 取最后两级）
    path_parts = os.path.normpath(seg_dir_path).split(os.sep)
    if len(path_parts) < 2:
        raise ValueError(f"seg目录路径格式异常，无法提取category和instance：{seg_dir_path}")
    instance_name = path_parts[-1]
    category_name = path_parts[-2]
    
    # 构建数据集内的Scan目录路径
    scan_dir_path = os.path.join(DATASET_ROOT, category_name, instance_name, SCAN_DIR_NAME)
    model_obj_path = os.path.join(scan_dir_path, TARGET_OBJ_NAME)
    point_cloud_path = os.path.join(scan_dir_path, TARGET_POINT_NAME)
    
    # 路径校验
    if not os.path.exists(scan_dir_path):
        raise FileNotFoundError(f"Scan目录不存在：{scan_dir_path}")
    if not os.path.exists(model_obj_path):
        raise FileNotFoundError(f"未找到目标obj文件：{model_obj_path}")
    if not os.path.exists(point_cloud_path):
        raise FileNotFoundError(f"未找到目标点云文件：{point_cloud_path}")

    # 加载模型、点云、标签（保持原逻辑，仅更新路径）
    try:
        model = trimesh.load(model_obj_path)
        if not isinstance(model, trimesh.Trimesh):
            raise ValueError("模型非Trimesh格式")
    except Exception as e:
        raise Exception(f"模型加载失败：{str(e)}")

    try:
        points = torch.load(point_cloud_path).numpy().astype(np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"点云形状异常（需(N,3)，实际{points.shape}）")
    except Exception as e:
        raise Exception(f"点云加载失败：{str(e)}")

    try:
        point_labels = np.loadtxt(seg_txt_path, dtype=int, delimiter=None).squeeze()
        if len(point_labels) != len(points):
            raise ValueError(f"标签数不匹配（标签{len(point_labels)} vs 点云{len(points)}）")
    except Exception as e:
        raise Exception(f"标签加载失败：{str(e)}")

    tqdm.write(f"✅ 加载完成：面数{len(model.faces)} | 点数{len(points)} | 模型路径{model_obj_path}")
    return model, points, point_labels


# -------------------------- 无修改函数（保持原逻辑） --------------------------
def build_face_adjacency(model):
    vertex_to_faces = defaultdict(list)
    for face_idx, face in enumerate(model.faces):
        for v in face:
            vertex_to_faces[v].append(face_idx)

    adjacency = [[] for _ in range(len(model.faces))]
    for face_idx, face in enumerate(model.faces):
        neighbor_candidates = set()
        for v in face:
            neighbor_candidates.update(vertex_to_faces[v])
        for neighbor in neighbor_candidates:
            if neighbor != face_idx and len(set(face) & set(model.faces[neighbor])) >= 2:
                adjacency[face_idx].append(neighbor)
    return adjacency


def preprocess_faces(model):
    faces = model.faces
    vertices = np.asarray(model.vertices, dtype=np.float64)
    num_faces = len(faces)
    
    face_bboxes = np.zeros((num_faces, 2, 3), dtype=np.float64)
    face_planes = np.zeros((num_faces, 4), dtype=np.float64)
    face_vertices = np.zeros((num_faces, 3, 3), dtype=np.float64)
    face_centers = np.zeros((num_faces, 3), dtype=np.float64)
    
    for i in range(num_faces):
        v = vertices[faces[i]]
        face_vertices[i] = v
        face_bboxes[i, 0] = v.min(axis=0)
        face_bboxes[i, 1] = v.max(axis=0)
        face_centers[i] = v.mean(axis=0)
        
        v0, v1, v2 = v
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        d = -np.dot(normal, v0)
        face_planes[i] = [normal[0], normal[1], normal[2], d]
    
    return face_vertices, face_bboxes, face_planes, face_centers


def vectorized_point_to_face_distance(points, face_vertices, face_planes):
    a, b, c, d = face_planes
    v0, v1, v2 = face_vertices
    
    plane_dist = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = points - v0
    
    dot00 = np.sum(v0v2 * v0v2)
    dot01 = np.sum(v0v2 * v0v1)
    dot02 = np.sum(v0p * v0v2, axis=1)
    dot11 = np.sum(v0v1 * v0v1)
    dot12 = np.sum(v0p * v0v1, axis=1)
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    in_triangle = (u >= 0) & (v >= 0) & (u + v <= 1)
    
    dist_to_v0 = np.linalg.norm(points - v0, axis=1)
    dist_to_v1 = np.linalg.norm(points - v1, axis=1)
    dist_to_v2 = np.linalg.norm(points - v2, axis=1)
    vertex_dist = np.minimum(np.minimum(dist_to_v0, dist_to_v1), dist_to_v2)
    
    return np.where(in_triangle, plane_dist, vertex_dist)


def assign_face_labels(model, points, point_labels, file_tqdm):
    face_vertices, face_bboxes, face_planes, face_centers = preprocess_faces(model)
    num_points = len(points)
    num_faces = len(model.faces)
    
    closest_face_indices = np.zeros(num_points, dtype=int)
    min_distances = np.full(num_points, np.inf, dtype=np.float64)
    
    num_batches = (num_faces + FACE_BATCH_SIZE - 1) // FACE_BATCH_SIZE
    batch_tqdm = tqdm(range(num_batches), desc="面匹配", leave=False, unit="批")
    
    for batch_idx in batch_tqdm:
        start = batch_idx * FACE_BATCH_SIZE
        end = min((batch_idx + 1) * FACE_BATCH_SIZE, num_faces)
        
        for face_idx in range(start, end):
            bbox_min, bbox_max = face_bboxes[face_idx]
            in_bbox = np.all((points >= bbox_min - 1e-3) & (points <= bbox_max + 1e-3), axis=1)
            if not np.any(in_bbox):
                continue
            
            points_in_bbox = points[in_bbox]
            distances = vectorized_point_to_face_distance(
                points_in_bbox, face_vertices[face_idx], face_planes[face_idx]
            )
            
            bbox_point_indices = np.where(in_bbox)[0]
            for i in range(len(points_in_bbox)):
                global_idx = bbox_point_indices[i]
                if distances[i] < min_distances[global_idx]:
                    min_distances[global_idx] = distances[i]
                    closest_face_indices[global_idx] = face_idx
        
        batch_tqdm.set_postfix({"已处理面": f"{end}/{num_faces}"})
    
    face_point_labels = defaultdict(list)
    for point_idx, face_idx in enumerate(closest_face_indices):
        face_point_labels[face_idx].append(point_labels[point_idx].item())
    
    face_labels = np.full(num_faces, -1, dtype=int)
    for face_idx, labels in face_point_labels.items():
        if labels:
            face_labels[face_idx] = Counter(labels).most_common(1)[0][0]
    
    assigned_count = np.sum(face_labels != -1)
    file_tqdm.write(f"🔍 面标签初步分配：{assigned_count}/{num_faces}个面已标记")
    return face_labels, face_centers


def fill_unlabeled_faces(face_labels, adjacency, face_centers, file_tqdm):
    initial_unlabeled = np.sum(face_labels == -1)
    if initial_unlabeled == 0:
        file_tqdm.write("✅ 所有面已标记，无需填充")
        return face_labels
    
    fill_tqdm = tqdm(range(MAX_FILL_ITER), desc="填充未标记面", leave=False, unit="轮")
    for iter_idx in fill_tqdm:
        new_labels = face_labels.copy()
        unlabeled_faces = np.where(face_labels == -1)[0]
        labeled_faces = np.where(face_labels != -1)[0]
        if len(labeled_faces) == 0:
            break
        
        updated = 0
        for face_idx in unlabeled_faces:
            neighbor_labels = [face_labels[n] for n in adjacency[face_idx] if face_labels[n] != -1]
            if neighbor_labels:
                new_labels[face_idx] = Counter(neighbor_labels).most_common(1)[0][0]
                updated += 1
        
        remaining_unlabeled = np.where(new_labels == -1)[0]
        if len(remaining_unlabeled) > 0:
            for face_idx in remaining_unlabeled:
                dists = np.linalg.norm(face_centers[labeled_faces] - face_centers[face_idx], axis=1)
                closest_idx = labeled_faces[np.argmin(dists)]
                new_labels[face_idx] = face_labels[closest_idx]
                updated += 1
        
        face_labels = new_labels
        remaining = np.sum(face_labels == -1)
        fill_tqdm.set_postfix({"剩余未标记面": remaining})
        if remaining == 0:
            break
    
    final_unlabeled = np.sum(face_labels == -1)
    if final_unlabeled > 0:
        file_tqdm.write(f"⚠️  仍有{final_unlabeled}个面未标记，用默认标签0填充")
        face_labels[face_labels == -1] = 0
    
    return face_labels


# -------------------------- 颜色映射函数（修复target_label+1的bug） --------------------------
def create_diverging_colormap(num_colors, h_neg=224, h_pos=28, as_cmap=False):
    """使用seaborn创建发散颜色映射"""
    cmap = sns.diverging_palette(
        h_neg=h_neg, h_pos=h_pos, s=75, l=60,
        n=num_colors, center="dark", as_cmap=as_cmap
    )
    
    if as_cmap:
        return cmap
    else:
        colors = []
        for i in range(num_colors):
            color = cmap[i][:3]
            rgb_color = [int(c * 255) for c in color]
            colors.append(rgb_color)
        return colors

def get_single_target_color(target_label):
    """为单个目标语义生成专属颜色（避免key错误）"""
    # 生成1种颜色即可（仅用于当前语义）
    colors = create_diverging_colormap(1, h_neg=224, h_pos=28)
    return colors[0]


# -------------------------- 新增：生成MTL材质文件（支持双材质） --------------------------
def write_mtl_file(mtl_path, target_label, target_color):
    """
    生成MTL文件：包含目标语义材质 + others材质
    mtl_path：mtl文件保存路径
    target_label：当前语义标签（如back）
    target_color：当前语义的RGB颜色（0-255）
    """
    # 材质名称（避免特殊字符）
    target_mat_name = f"{MATERIAL_NAME_PREFIX}{target_label}"
    others_mat_name = f"{MATERIAL_NAME_PREFIX}{OTHERS_LABEL}"
    
    with open(mtl_path, "w", encoding="utf-8") as f:
        # 1. 目标语义材质（专属颜色）
        f.write(f"newmtl {target_mat_name}\n")
        f.write("Ns 100.000000\n")       # 高光强度
        f.write("Ka 1.000000 1.000000 1.000000\n")  # 环境光
        f.write("Kd %.6f %.6f %.6f\n" % (target_color[0]/255, target_color[1]/255, target_color[2]/255))  # 漫反射颜色
        f.write("Ks 0.500000 0.500000 0.500000\n")  # 高光颜色
        f.write("Ke 0.000000 0.000000 0.000000\n")  # 自发光
        f.write("Ni 1.450000\n")        # 折射率
        f.write("d 1.000000\n")         # 透明度
        f.write("illum 2\n\n")          # 光照模型（Phong）
        
        # 2. Others材质（灰色）
        f.write(f"newmtl {others_mat_name}\n")
        f.write("Ns 100.000000\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd %.6f %.6f %.6f\n" % (OTHERS_COLOR[0]/255, OTHERS_COLOR[1]/255, OTHERS_COLOR[2]/255))
        f.write("Ks 0.500000 0.500000 0.500000\n")
        f.write("Ke 0.000000 0.000000 0.000000\n")
        f.write("Ni 1.450000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")


# -------------------------- 核心修改：手动编写OBJ+MTL文件（替换trimesh.export） --------------------------
def save_batch_results(model, face_labels, seg_dir_path):
    """
    保存结果：每个语义单独生成 obj+mtl，仅包含两种颜色
    保存结构：
    {instance}/seg_obj/
    ├── label_0/          # 语义标签0（如back）
    │   ├── label_0.obj   # 包含标签0的面片（专属色）+ 其他面片（灰色）
    │   └── label_0.mtl   # 对应两种材质
    ├── label_1/          # 语义标签1（如leg）
    │   ├── label_1.obj
    │   └── label_1.mtl
    └── face_labels.txt   # 所有面的原始标签（保留）
    """
    output_root = os.path.join(seg_dir_path, OUTPUT_FOLDER_NAME)
    os.makedirs(output_root, exist_ok=True)
    
    # 1. 保存原始面标签（保留原功能，便于溯源）
    face_label_path = os.path.join(output_root, "face_labels.txt")
    np.savetxt(face_label_path, face_labels, fmt="%d", delimiter="\n")
    
    # 2. 获取所有独特语义标签（去重）
    unique_labels = sorted(np.unique(face_labels))
    tqdm.write(f"📌 该模型包含语义标签：{unique_labels}")
    
    # 3. 为每个语义标签生成独立的obj+mtl
    for target_label in tqdm(unique_labels, desc="生成单语义OBJ+MTL", leave=False):
        # 3.1 创建语义专属文件夹（避免文件冲突）
        label_dir = os.path.join(output_root, f"label_{target_label}")
        os.makedirs(label_dir, exist_ok=True)
        
        # 3.2 定义文件路径
        obj_path = os.path.join(label_dir, f"label_{target_label}.obj")
        mtl_path = os.path.join(label_dir, f"label_{target_label}.mtl")
        mtl_filename = os.path.basename(mtl_path)  # OBJ中引用MTL用的文件名
        
        # 3.3 生成当前语义的专属颜色并写入MTL文件（修复bug，不再用target_label+1）
        target_color = get_single_target_color(target_label)
        write_mtl_file(mtl_path, target_label, target_color)
        
        # 3.4 提取模型顶点和面（trimesh格式转numpy）
        vertices = np.asarray(model.vertices, dtype=np.float64)  # (V, 3)
        faces = np.asarray(model.faces, dtype=np.int32)          # (F, 3)
        num_faces = len(faces)
        
        # 3.5 手动写入OBJ文件（参考用户提供的示例逻辑）
        with open(obj_path, "w", encoding="utf-8") as f_obj:
            # 第一步：引用MTL文件
            f_obj.write(f"mtllib {mtl_filename}\n\n")
            
            # 第二步：写入顶点坐标（保留6位小数）
            for v in vertices:
                f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f_obj.write("\n")
            
            # 第三步：按材质分组写入面（OBJ面索引从1开始，需+1）
            target_mat_name = f"{MATERIAL_NAME_PREFIX}{target_label}"
            others_mat_name = f"{MATERIAL_NAME_PREFIX}{OTHERS_LABEL}"
            
            # 先写目标语义的面
            f_obj.write(f"usemtl {target_mat_name}\n")
            target_face_indices = np.where(face_labels == target_label)[0]
            for idx in target_face_indices:
                face = faces[idx]
                f_obj.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            # 再写others的面
            f_obj.write(f"\nusemtl {others_mat_name}\n")
            others_face_indices = np.where(face_labels != target_label)[0]
            for idx in others_face_indices:
                face = faces[idx]
                f_obj.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        tqdm.write(f"💾 保存完成：{os.path.basename(label_dir)}/ → obj+mtl")


# -------------------------- 主流程（适配新路径，简化参数传递） --------------------------
def main():
    print("="*60)
    print("🚀 开始批量处理：单语义+others双标签OBJ+MTL生成")
    print(f"📁 seg.txt搜索根目录：{SEG_TXT_SEARCH_ROOT}")
    print(f"📁 数据集根目录：{DATASET_ROOT}")
    print(f"🔍 模型路径规则：{DATASET_ROOT}/{{category}}/{{instance}}/{SCAN_DIR_NAME}/{TARGET_OBJ_NAME}")
    print(f"🔍 点云路径规则：{DATASET_ROOT}/{{category}}/{{instance}}/{SCAN_DIR_NAME}/{TARGET_POINT_NAME}")
    print(f"💾 结果保存位置：{{instance}}/{OUTPUT_FOLDER_NAME}/（每个语义单独文件夹）")
    print("📌 核心规则：每个OBJ仅含两种颜色→当前语义（专属色）+ 其他（灰色）")
    print("="*60)

    # 1. 搜索所有seg.txt文件（递归遍历）
    try:
        if not os.path.exists(SEG_TXT_SEARCH_ROOT):
            raise FileNotFoundError(f"seg.txt搜索根目录不存在：{SEG_TXT_SEARCH_ROOT}")
        
        task_list = []
        for root, _, files in tqdm(os.walk(SEG_TXT_SEARCH_ROOT), desc="搜索seg.txt文件", unit="目录"):
            if "seg.txt" in files:
                # 保存seg.txt所在目录（用于解析category和instance）
                task_list.append(root)
        
        if not task_list:
            raise ValueError(f"未在{SEG_TXT_SEARCH_ROOT}下找到任何seg.txt")
        print(f"📊 搜索完成：共{len(task_list)}个有效seg.txt文件待处理")
    
    except Exception as e:
        print(f"❌ seg.txt搜索失败：{str(e)}")
        return

    # 2. 批量处理每个模型
    global_tqdm = tqdm(task_list, desc="全局处理进度", unit="模型")
    success_count = 0
    fail_count = 0

    for seg_dir_path in global_tqdm:
        # 提取层级信息（category name / instance name）
        path_parts = os.path.normpath(seg_dir_path).split(os.sep)
        instance_name = path_parts[-1]
        category_name = path_parts[-2]
        global_tqdm.set_postfix({"类别": category_name, "实例": instance_name})

        try:
            # 核心流程：加载数据→构建邻接表→分配标签→填充→保存（适配新路径）
            model, points, point_labels = load_batch_data(seg_dir_path)
            adjacency = build_face_adjacency(model)
            face_labels, face_centers = assign_face_labels(model, points, point_labels, global_tqdm)
            face_labels = fill_unlabeled_faces(face_labels, adjacency, face_centers, global_tqdm)
            
            # 模型旋转（保留原逻辑，可选调整）
            ROTATION_MATRIX = trimesh.transformations.rotation_matrix(
                angle=-np.pi / 2,
                direction=[1, 0, 0]
            ) 
            model.apply_transform(ROTATION_MATRIX)
            
            # 保存结果
            save_batch_results(model, face_labels, seg_dir_path)
            
            success_count += 1
            global_tqdm.write(f"🎉 处理成功（{success_count}/{len(task_list)}）：{category_name}/{instance_name}")

        except Exception as e:
            fail_count += 1
            global_tqdm.write(f"❌ 处理失败（{fail_count}/{len(task_list)}）：{str(e)}")
            continue

    # 3. 输出最终统计
    print("\n" + "="*60)
    print("📊 批量处理完成！")
    print(f"✅ 成功：{success_count}个模型")
    print(f"❌ 失败：{fail_count}个模型")
    print(f"📌 结果特点：每个语义单独OBJ+MTL，仅含「当前语义色+others灰色」")
    print(f"📌 结果根目录：{SEG_TXT_SEARCH_ROOT}（每个Instance目录下的{OUTPUT_FOLDER_NAME}文件夹）")
    print("="*60)


if __name__ == "__main__":
    # 依赖库检查（保持原逻辑）
    required_libs = ["trimesh", "torch", "matplotlib", "seaborn"]
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    if missing_libs:
        print(f"❌ 缺少依赖库：{', '.join(missing_libs)}，请执行安装：")
        print(f"pip install {' '.join(missing_libs)}")
    else:
        main()