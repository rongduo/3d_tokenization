"""
    确定训练数据，是否都在规范空间中
    python -m release_pipeline2.remark.a6_check

    另一个，确认lid 和 底 text的相似度

    顺便测试一下，如果用 qwen embedding 特征会不会更好一些 ： https://huggingface.co/Qwen/Qwen3-Embedding-8B
    再则做一下和clip文本相似度的对比


    all-MiniLM-L6-v2模型效果会更好？

    all-mpnet-base-v2模型效果会更好？
"""


############# all-mpnet-base-v2模型效果会更好？
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer, AutoModel

# 加载模型和分词器（替换为all-mpnet-base-v2，输出768维特征）
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 自定义颜色映射（与原测试保持一致）
cmap = LinearSegmentedColormap.from_list("sim_cmap", ["#4169e1", "#ffffff", "#cd5c5c"])

def mean_pooling(model_output, attention_mask):
    """MPNet专用平均池化（与原逻辑一致，适配768维特征）"""
    token_embeddings = model_output[0]  # 最后一层隐藏状态 [batch_size, seq_len, 768]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

def get_text_embeddings(texts):
    """批量生成768维文本嵌入"""
    # 分词（MPNet推荐最大长度为384，比MiniLM更长）
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=384,  # 适配MPNet的最佳长度
        return_tensors="pt"
    )
    
    # 生成嵌入
    with torch.no_grad():
        model_output = model(** inputs)
    
    # 平均池化并归一化（保持与原测试的特征处理一致）
    embeddings = mean_pooling(model_output, inputs["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2归一化
    return embeddings

def cosine_similarity_matrix(embeddings):
    """计算768维特征的两两余弦相似度矩阵"""
    return torch.mm(embeddings, embeddings.T).numpy()  # 矩阵乘法等价于余弦相似度（因已归一化）


def calculate_pairwise_similarity(texts, save_heatmap=False, save_path=None):
    """计算文本两两相似度（逻辑与原测试完全一致）"""
    embeddings = get_text_embeddings(texts)
    print('embeddings:', embeddings.shape)
    sim_matrix = cosine_similarity_matrix(embeddings)
    
    # 打印结果（验证768维特征的相似度表现）
    print("文本列表:", texts)
    print("特征维度:", embeddings.shape[1])  # 确认输出为768维
    print("两两相似度矩阵:")
    for i, text_i in enumerate(texts):
        row = [f"{sim_matrix[i, j]:.2f}" for j, text_j in enumerate(texts)]
        print(f"{text_i}: {row}")

    
    return texts, sim_matrix

# 示例使用（与原测试用例保持一致，便于对比）
if __name__ == "__main__":

    # 额外测试更细粒度的物体部件名称
    # sample_texts = ['bottom_panel', 'lid', 'bottle_body', 'bottle_neck']
    # sample_texts = ['arm', 'back', 'leg', 'seat', 'wheel']
    # sample_texts = ['lid', 'lid of a bottle', 'bottle_body', 'bottom_panel']
    sample_texts = ['bottom_panel', 'non-bottom_panel', 'lid', 'non-lid', 'bottle_body', 'non-bottle_body', 'bottle_neck', 'non-bottle_neck']
    
    calculate_pairwise_similarity(
        texts=sample_texts,
        save_heatmap=True,
        save_path="mpnet_similarity_heatmap.png"
    )



asdf
############## all-MiniLM-L6-v2模型效果会更好？
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer, AutoModel

# 加载模型和分词器
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 自定义颜色映射（蓝色低相似度，红色高相似度）
cmap = LinearSegmentedColormap.from_list("sim_cmap", ["#4169e1", "#ffffff", "#cd5c5c"])

def mean_pooling(model_output, attention_mask):
    """对模型输出进行平均池化，生成句子嵌入"""
    token_embeddings = model_output[0]  # 取最后一层隐藏状态 [batch_size, seq_len, hidden_size]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

def get_text_embeddings(texts):
    """批量生成文本嵌入"""
    # 分词
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,  # 模型支持的最大长度
        return_tensors="pt"
    )
    
    # 生成嵌入
    with torch.no_grad():
        model_output = model(** inputs)
    
    # 平均池化并归一化
    embeddings = mean_pooling(model_output, inputs["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def cosine_similarity_matrix(embeddings):
    """计算嵌入矩阵的两两余弦相似度矩阵"""
    return torch.mm(embeddings, embeddings.T).numpy()

def plot_similarity_heatmap(sim_matrix, labels, save_path=None):
    """绘制相似度热力图"""
    plt.figure(figsize=(len(labels)*0.8, len(labels)*0.8))
    plt.imshow(sim_matrix, cmap=cmap, vmin=0, vmax=1)  # all-MiniLM相似度通常在0~1之间
    
    # 添加标签和标题
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title("文本两两相似度热力图 (all-MiniLM-L6-v2)", fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label("余弦相似度 (0 ~ 1)", fontsize=8)
    
    # 标注相似度值
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}", 
                     ha="center", va="center", fontsize=6,
                     color="black" if 0.3 < sim_matrix[i, j] < 0.7 else "white")
    
    # plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #     print(f"热力图已保存至: {save_path}")
    # plt.close()

def calculate_pairwise_similarity(texts, save_heatmap=False, save_path=None):
    """
    计算文本列表的两两相似度
    texts: 待比较的文本列表（如["lid", "bottom", "handle"]）
    save_heatmap: 是否保存热力图
    save_path: 热力图保存路径（若save_heatmap=True）
    返回: (标签列表, 相似度矩阵)
    """
    # 生成嵌入
    embeddings = get_text_embeddings(texts)
    
    # 计算相似度矩阵
    sim_matrix = cosine_similarity_matrix(embeddings)
    
    # 打印结果
    print("文本列表:", texts)
    print("两两相似度矩阵:")
    for i, text_i in enumerate(texts):
        row = [f"{sim_matrix[i, j]:.2f}" for j, text_j in enumerate(texts)]
        print(f"{text_i}: {row}")
    
    # # 保存热力图
    # if save_heatmap and save_path:
    #     plot_similarity_heatmap(sim_matrix, texts, save_path)
    
    return texts, sim_matrix

# 示例使用
if __name__ == "__main__":
    # 待比较的语义单词（可替换为实际场景的单词）
    sample_texts = [
        "lid", "bottom", "handle", 
        "top cover", "base", "grip"
    ]
    sample_texts = ['bottom_panel', 'lid', 'bottle_body', 'bottle_neck']
    
    # 计算相似度
    calculate_pairwise_similarity(
        texts=sample_texts,
        save_heatmap=True,
        save_path="miniLM_similarity_heatmap.png"
    )

asdf
########### qwen embedding 提取文本特征，会不会对比更明显
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel


# 加载Qwen3-Embedding模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Embedding-8B")
# 加载clip-Embedding模型和分词器
# 全局加载SigLIP模型和分词器（适配模型最大序列长度）
siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
# 获取模型允许的最大序列长度（从配置中读取，通常为64）
max_seq_length = 64 

# 自定义颜色映射（蓝色表示低相似度，红色表示高相似度）
cmap = LinearSegmentedColormap.from_list("sim_cmap", ["#4169e1", "#ffffff", "#cd5c5c"])

'''def get_text_embedding(text):
    """使用Qwen3-Embedding模型生成文本嵌入"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # 取最后一层隐藏状态的均值作为嵌入
    hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
    embedding = torch.mean(hidden_states, dim=1).squeeze()  # 平均池化得到句子嵌入
    return embedding'''
def get_text_embedding(text):
    """使用SigLIP模型生成文本嵌入（修正配置属性错误）"""
    if not isinstance(text, str):
        raise TypeError(f"输入必须为字符串，实际为: {type(text)}")
    
    # 使用模型支持的最大序列长度64
    inputs = siglip_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,  # 明确设置为64
        return_tensors="pt"
    )
    
    with torch.no_grad():
        text_feat = siglip_model.get_text_features(** inputs)  # 生成特征
    
    # 归一化并去除批量维度
    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    return text_feat.squeeze(0)


def load_labels(mask_labels_path):
    """加载标签列表"""
    if not os.path.exists(mask_labels_path):
        raise FileNotFoundError(f"标签文件不存在: {mask_labels_path}")
    with open(mask_labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError(f"标签文件为空: {mask_labels_path}")
    return labels

def cosine_similarity_matrix(embeddings):
    """计算嵌入矩阵的两两余弦相似度矩阵"""
    # 归一化嵌入（每行一个嵌入向量）
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    normalized_embeds = embeddings / (norms + 1e-8)
    # 相似度矩阵 = 归一化嵌入 × 归一化嵌入的转置
    sim_matrix = torch.mm(normalized_embeds, normalized_embeds.T)
    return sim_matrix.numpy()

def plot_similarity_heatmap(sim_matrix, labels, save_path=None):
    """绘制相似度热力图"""
    plt.figure(figsize=(len(labels)*0.8, len(labels)*0.8))
    plt.imshow(sim_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # 添加标签和标题
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title("标签文本两两相似度热力图（Qwen3-Embedding）", fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label("余弦相似度 (-1 ~ 1)", fontsize=8)
    
    # 在热力图上标注相似度值（保留2位小数）
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}", 
                     ha="center", va="center", fontsize=6,
                     color="black" if 0.3 < sim_matrix[i, j] < 0.7 else "white")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"热力图已保存至: {save_path}")
    plt.close()

def check_all_pair_similarity(data_dir, save_heatmap=True):
    """
    计算指定目录下所有标签的两两文本相似度（使用Qwen3-Embedding）
    save_heatmap: 是否保存热力图
    返回: (标签列表, 相似度矩阵)
    """
    # 构建文件路径
    mask_labels_path = os.path.join(data_dir, "mask_labels.txt")
    
    # 加载标签并生成嵌入
    labels = load_labels(mask_labels_path)
    num_labels = len(labels)
    print(f"生成 {num_labels} 个标签的嵌入...")
    labels = ["arm",
        "back",
        "leg",
        "seat",
        "wheel"]
    
    # 为每个标签生成嵌入
    embeddings = []
    for label in labels:
        embed = get_text_embedding(label)
        embeddings.append(embed)
    embeddings = torch.stack(embeddings)  # 形状: (num_labels, embed_dim)
    
    # 计算相似度矩阵
    sim_matrix = cosine_similarity_matrix(embeddings)
    
    # 打印相似度矩阵（文本形式）
    print(f"\n===== 数据目录: {data_dir} =====")
    print(f"标签列表: {labels}")
    print("两两相似度矩阵:")
    for i in range(num_labels):
        row = [f"{sim_matrix[i, j]:.2f}" for j in range(num_labels)]
        print(f"标签 {i} ({labels[i]}): {row}")
    
    # # 保存热力图
    # if save_heatmap:
    #     heatmap_save_path = os.path.join(data_dir, "label_similarity_heatmap_qwen.png")
    #     plot_similarity_heatmap(sim_matrix, labels, heatmap_save_path)
    
    return labels, sim_matrix

# 示例使用
if __name__ == "__main__":
    # 可替换为需要分析的目录列表
    data_dirs = [
        "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtrain/fine_b'32_056'",
        # 添加更多目录...
    ]
    
    for dir_path in data_dirs:

        check_all_pair_similarity(dir_path, save_heatmap=True)



asdf
####### 相似度check
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 自定义颜色映射（蓝色表示低相似度，红色表示高相似度）
cmap = LinearSegmentedColormap.from_list("sim_cmap", ["#4169e1", "#ffffff", "#cd5c5c"])

def load_text_features(text_feat_path):
    """加载文本特征文件并验证格式"""
    if not os.path.exists(text_feat_path):
        raise FileNotFoundError(f"文本特征文件不存在: {text_feat_path}")
    try:
        text_feats = torch.load(text_feat_path, map_location="cpu")
        if not isinstance(text_feats, torch.Tensor):
            raise TypeError(f"文本特征应为张量，实际为: {type(text_feats)}")
        if len(text_feats.shape) != 2:
            raise ValueError(f"文本特征应为2D张量 (num_labels, feat_dim)，实际形状: {text_feats.shape}")
        return text_feats
    except Exception as e:
        raise RuntimeError(f"加载文本特征失败: {str(e)}")

def load_labels(mask_labels_path):
    """加载标签列表"""
    if not os.path.exists(mask_labels_path):
        raise FileNotFoundError(f"标签文件不存在: {mask_labels_path}")
    with open(mask_labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError(f"标签文件为空: {mask_labels_path}")
    return labels

def cosine_similarity_matrix(features):
    """计算特征矩阵的两两余弦相似度矩阵"""
    # 归一化特征（每行一个特征向量）
    norms = torch.norm(features, dim=1, keepdim=True)
    normalized_feats = features / (norms + 1e-8)
    # 相似度矩阵 = 归一化特征 × 归一化特征的转置
    sim_matrix = torch.mm(normalized_feats, normalized_feats.T)
    return sim_matrix.numpy()

def plot_similarity_heatmap(sim_matrix, labels, save_path=None):
    """绘制相似度热力图"""
    plt.figure(figsize=(len(labels)*0.8, len(labels)*0.8))
    plt.imshow(sim_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # 添加标签和标题
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title("标签文本特征两两相似度热力图", fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label("余弦相似度 (-1 ~ 1)", fontsize=8)
    
    # 在热力图上标注相似度值（保留2位小数）
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}", 
                     ha="center", va="center", fontsize=6,
                     color="black" if 0.3 < sim_matrix[i, j] < 0.7 else "white")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"热力图已保存至: {save_path}")
    plt.close()

def check_all_pair_similarity(data_dir, save_heatmap=True):
    """
    计算指定目录下所有标签的两两文本特征相似度
    save_heatmap: 是否保存热力图
    返回: (标签列表, 相似度矩阵)
    """
    # 构建文件路径
    mask_labels_path = os.path.join(data_dir, "mask_labels.txt")
    text_feat_path = os.path.join(data_dir, "text_feat.pt")
    
    # 加载数据
    labels = load_labels(mask_labels_path)
    text_feats = load_text_features(text_feat_path)
    num_labels = len(labels)
    
    # 检查特征数量与标签数量是否匹配
    if text_feats.shape[0] != num_labels:
        raise ValueError(f"文本特征数量 ({text_feats.shape[0]}) 与标签数量 ({num_labels}) 不匹配")
    
    # 计算相似度矩阵
    sim_matrix = cosine_similarity_matrix(text_feats)
    
    # 打印相似度矩阵（文本形式）
    print(f"\n===== 数据目录: {data_dir} =====")
    print(f"标签列表: {labels}")
    print("两两相似度矩阵:")
    for i in range(num_labels):
        row = [f"{sim_matrix[i, j]:.2f}" for j in range(num_labels)]
        print(f"标签 {i} ({labels[i]}): {row}")
    
    # # 保存热力图
    # if save_heatmap:
    #     heatmap_save_path = os.path.join(data_dir, "label_similarity_heatmap.png")
    #     plot_similarity_heatmap(sim_matrix, labels, heatmap_save_path)
    
    return labels, sim_matrix

# 示例使用
if __name__ == "__main__":
    # 可替换为需要分析的目录列表
    data_dirs = [
        "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtrain/fine_b'32_056'",
        # 添加更多目录...
    ]
    
    for dir_path in data_dirs:
        try:
            check_all_pair_similarity(dir_path, save_heatmap=True)
        except Exception as e:
            print(f"处理目录失败 {dir_path}: {str(e)}")






asdf
####### 确定训练数据，是否都在规范空间中
import torch
import os
import open3d as o3d
import numpy as np

# 配置路径
data_root = '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/bottle'
save_dir = 'results_tmp/oracheck'
os.makedirs(save_dir, exist_ok=True)

def get_direction_vector(pts_xyz, lid_mask):
    """
    计算lid的方向向量：lid平均点 减去 整体点云平均点
    返回归一化后的方向向量
    """
    # 整体点云的平均点
    all_mean = pts_xyz.mean(dim=0)
    # lid点的平均点
    lid_pts = pts_xyz[lid_mask]
    lid_mean = lid_pts.mean(dim=0)
    # 方向向量（lid平均点 - 整体平均点）
    direction = lid_mean - all_mean
    # 归一化
    direction_norm = direction / (torch.norm(direction) + 1e-8)
    return direction_norm

def is_same_direction(vec1, vec2, threshold=0.9):
    """判断两个方向向量是否同向（余弦相似度 >= threshold）"""
    cos_sim = torch.dot(vec1, vec2)
    return cos_sim >= threshold

def save_point_cloud(xyz, rgb, save_path):
    """保存点云为PLY文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgb.numpy() / 255.0)  # 颜色归一化到0-1
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"已保存点云至: {save_path}")

# 读取物体路径列表
with open(f"{data_root}/train.txt", "r") as f:
    obj_path_list = [line.strip() for line in f if line.strip()]

if not obj_path_list:
    raise ValueError("train.txt中未找到有效路径")

# 存储结果的列表
same_direction = []  # 朝向一致的物体 (data_dir, xyz, rgb, direction)
diff_direction = []   # 朝向不一致的物体 (data_dir, xyz, rgb, direction)
first_direction = None  # 基准方向向量
first_xyz = None
first_rgb = None

# 遍历所有物体，计算方向并分类
for idx, data_dir in enumerate(obj_path_list):
    # 检查必要文件
    required_files = [
        "mask2points.pt", "normals_transformed.pt", 
        "mask_labels.txt", "points_transformed.pt", "rgb.pt"
    ]
    file_paths = [os.path.join(data_dir, f) for f in required_files]
    if not all(os.path.exists(p) for p in file_paths):
        print(f"警告：{data_dir} 缺少文件，跳过")
        continue

    # 加载数据
    with open(file_paths[2], "r") as f:
        labels = f.read().splitlines()
    mask_pts = torch.load(file_paths[0], map_location="cpu")  # [num_labels, num_points]
    normal = torch.load(file_paths[1], map_location="cpu")     # [num_points, 3]
    pts_xyz = torch.load(file_paths[3], map_location="cpu")    # [num_points, 3]
    pts_rgb = torch.load(file_paths[4], map_location="cpu") * 255  # [num_points, 3]

    # 检查lid标签
    if "lid" not in labels:
        print(f"警告：{data_dir} 无lid标签，跳过")
        continue
    lid_idx = labels.index("lid")
    lid_mask = mask_pts[lid_idx] == 1  # lid对应的点掩码

    # 检查lid点是否有效
    if lid_mask.sum() == 0:
        print(f"警告：{data_dir} 无有效lid点，跳过")
        continue

    # 计算方向向量（lid平均点 相对 整体平均点的方向）
    direction = get_direction_vector(pts_xyz, lid_mask)

    # 处理第一个物体（基准）
    if idx == 0:
        first_direction = direction
        first_xyz = pts_xyz
        first_rgb = pts_rgb
        print(f"已设置基准物体：{data_dir}")
        continue

    # 判断与基准方向是否一致
    same = is_same_direction(direction, first_direction)
    if same:
        same_direction.append((data_dir, pts_xyz, pts_rgb, direction))
        print(f"物体 {data_dir} 与基准朝向一致（累计：{len(same_direction)}）")
    else:
        diff_direction.append((data_dir, pts_xyz, pts_rgb, direction))
        print(f"物体 {data_dir} 与基准朝向不一致（累计：{len(diff_direction)}）")

# 计算比例
total = len(same_direction) + len(diff_direction)
same_ratio = len(same_direction) / total if total > 0 else 0.0
diff_ratio = len(diff_direction) / total if total > 0 else 0.0

# 保存点云（基准+最多2个同向+最多2个不同向）
saved = 0
# 保存基准物体
if first_xyz is not None:
    save_path = os.path.join(save_dir, "0_baseline.ply")
    save_point_cloud(first_xyz, first_rgb, save_path)
    saved += 1

# 保存同向物体
save_same = min(2, len(same_direction))
for i in range(save_same):
    data_dir, xyz, rgb, _ = same_direction[i]
    name = os.path.basename(data_dir)
    save_path = os.path.join(save_dir, f"same_{i+1}_{name}.ply")
    save_point_cloud(xyz, rgb, save_path)
    saved += 1

# 保存不同向物体
save_diff = min(2, len(diff_direction))
for i in range(save_diff):
    data_dir, xyz, rgb, _ = diff_direction[i]
    name = os.path.basename(data_dir)
    save_path = os.path.join(save_dir, f"diff_{i+1}_{name}.ply")
    save_point_cloud(xyz, rgb, save_path)
    saved += 1

# 输出统计结果
print("\n" + "="*50)
print(f"基准物体方向向量：{first_direction.numpy()}")
print(f"总有效物体数：{total}（基准物体除外）")
print(f"同向物体数：{len(same_direction)}，占比：{same_ratio:.2%}")
print(f"不同向物体数：{len(diff_direction)}，占比：{diff_ratio:.2%}")
print(f"已保存 {saved} 个点云文件至 {save_dir}")
print(f" - 基准物体：1个")
print(f" - 同向物体：{save_same}个（共{len(same_direction)}个）")
print(f" - 不同向物体：{save_diff}个（共{len(diff_direction)}个）")
print("="*50)
