"""
对比三种 token 生成方法的速度和几何一致性。
1. Voxel 分组  2. FPS + k-NN  3. 法向超点 (region growing)
"""
import torch
import numpy as np
import time
import os, sys

# 确保能找到项目包
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

TRAIN_TXT = "/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat/train.txt"
SAMPLE_COUNT = 20  # 随机抽几个对象测试


def load_object(obj_dir: str):
    """加载一个对象的点云数据"""
    xyz = torch.load(os.path.join(obj_dir, "points.pt"), weights_only=True).numpy().astype(np.float32)
    rgb = torch.load(os.path.join(obj_dir, "rgb.pt"), weights_only=True).numpy().astype(np.float32)
    normal = torch.load(os.path.join(obj_dir, "normals.pt"), weights_only=True).numpy().astype(np.float32)
    mask_pts = torch.load(os.path.join(obj_dir, "mask2points.pt"), weights_only=True).numpy()
    # 读 ground truth part token（partsam 生成的）
    gt_tokens = torch.load(os.path.join(obj_dir, "parts.pt"), weights_only=True).numpy().reshape(-1)
    return xyz, rgb, normal, mask_pts, gt_tokens


# ============================================================
# 方法 1: Voxel 分组 —— 放大体素，用 grid_coord hash 做 token
# ============================================================
def tokenize_voxel(xyz: np.ndarray, voxel_size: float = 0.06):
    """
    对点云做体素化，每个非空体素 = 一个 token。
    grid_coord = floor(coord / voxel_size)，hash 编码为唯一 ID。
    """
    # 中心化
    xyz_c = xyz - xyz.min(axis=0)
    grid = np.floor(xyz_c / voxel_size).astype(np.int32)
    # hash: grid * large primes sum
    tokens = grid[:, 0] * 73856093 + grid[:, 1] * 19349663 + grid[:, 2] * 83492791
    # 压缩到 0..n_token-1
    _, tokens = np.unique(tokens, return_inverse=True)
    return tokens.astype(np.int64)


# ============================================================
# 方法 2: FPS + k-NN 分组
# ============================================================
def tokenize_fps_knn(xyz: np.ndarray, n_seeds: int = 100):
    """
    Farthest Point Sampling 选种子点 → 每个点分配给最近种子。
    """
    from sklearn.neighbors import NearestNeighbors

    N = xyz.shape[0]
    if n_seeds >= N:
        return np.arange(N, dtype=np.int64)

    # FPS
    pts = torch.from_numpy(xyz).float()
    seeds = [np.random.randint(N)]
    dist = torch.full((N,), 1e10)
    for _ in range(n_seeds - 1):
        d = ((pts - pts[seeds[-1]]) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        seeds.append(int(torch.argmax(dist)))

    seeds_xyz = xyz[seeds]
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(seeds_xyz)
    tokens = nn.kneighbors(xyz, return_distance=False).reshape(-1)
    return tokens.astype(np.int64)


# ============================================================
# 方法 3: 法向超点 —— 向量化实现
# ============================================================
def tokenize_superpoint(xyz: np.ndarray, normal: np.ndarray,
                        k: int = 15, angle_thresh_deg: float = 30.0,
                        min_size: int = 3):
    """
    向量化实现：scipy CSR 稀疏图 + connected_components。
    1) k-NN 建图  2) 法向角度阈值筛选边  3) 连通分量 = token
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors

    N = xyz.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xyz)
    _, idx = nn.kneighbors(xyz)
    idx = idx[:, 1:]  # 去掉自身 [N, k]

    # 法向归一化
    n_norm = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)

    # 向量化计算所有邻接边的法向夹角
    ni = n_norm[:, None, :]           # [N, 1, 3]
    nj = n_norm[idx]                   # [N, k, 3]
    cos_angles = np.abs(np.sum(ni * nj, axis=2))  # [N, k]
    valid_mask = cos_angles > np.cos(np.deg2rad(angle_thresh_deg))

    # 构建 CSR 稀疏邻接矩阵
    row_inds = np.repeat(np.arange(N), k)[valid_mask.ravel()]
    col_inds = idx.ravel()[valid_mask.ravel()]
    data = np.ones(len(row_inds), dtype=np.int8)
    adj_mat = csr_matrix((data, (row_inds, col_inds)), shape=(N, N))

    n_components, labels = connected_components(adj_mat, directed=False)

    # 过滤太小的 cluster：归并到最近的大 cluster
    sizes = np.bincount(labels)
    small = np.where(sizes < min_size)[0]
    big = np.where(sizes >= min_size)[0]
    if len(small) > 0 and len(big) > 0:
        big_centers = np.array([xyz[labels == b].mean(axis=0) for b in big])
        for s in small:
            mask = labels == s
            if mask.sum() == 0:
                continue
            sc = xyz[mask].mean(axis=0)
            nearest = big[np.argmin(np.linalg.norm(big_centers - sc, axis=1))]
            labels[mask] = nearest

    _, tokens = np.unique(labels, return_inverse=True)
    return tokens.astype(np.int64)


# ============================================================
# 对比指标
# ============================================================
def adjusted_rand_index(labels1, labels2):
    """Adjusted Rand Index — 衡量两种分组的相似度，[-1, 1]，越大越一致"""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels1, labels2)


def homogeneity_completeness(gt, pred):
    """同质性 V-measure"""
    from sklearn.metrics import homogeneity_score, completeness_score
    return homogeneity_score(gt, pred), completeness_score(gt, pred)


def n_tokens(tokens):
    return len(np.unique(tokens))


# ============================================================
# Main
# ============================================================
def main():
    with open(TRAIN_TXT) as f:
        all_dirs = [l.strip() for l in f if l.strip()]
    rng = np.random.RandomState(42)
    sample_dirs = rng.choice(all_dirs, min(SAMPLE_COUNT, len(all_dirs)), replace=False)

    results = {
        "voxel_5cm":    {"time": [], "n_tokens": [], "ari": []},
        "voxel_8cm":    {"time": [], "n_tokens": [], "ari": []},
        "fps_knn_gt":   {"time": [], "n_tokens": [], "ari": []},
        "fps_knn_50":   {"time": [], "n_tokens": [], "ari": []},
        "superpoint":   {"time": [], "n_tokens": [], "ari": []},
        "superp_lite":  {"time": [], "n_tokens": [], "ari": []},
    }

    for di, obj_dir in enumerate(sample_dirs):
        xyz, rgb, normal, mask_pts, gt_tokens = load_object(obj_dir)
        N = xyz.shape[0]
        gt_n = n_tokens(gt_tokens)

        name = os.path.basename(obj_dir)
        print(f"[{di+1}/{len(sample_dirs)}] {name}  N={N}  GT parts={gt_n}")

        # Voxel 5cm / 8cm
        for vs, key in [(0.05, "voxel_5cm"), (0.08, "voxel_8cm")]:
            t0 = time.time()
            tok = tokenize_voxel(xyz, voxel_size=vs)
            elapsed = time.time() - t0
            results[key]["time"].append(elapsed)
            results[key]["n_tokens"].append(n_tokens(tok))
            results[key]["ari"].append(adjusted_rand_index(gt_tokens, tok))
            print(f"  Voxel{vs*100:.0f}cm: {elapsed*1000:.1f}ms, {n_tokens(tok):4d} tok, ARI={results[key]['ari'][-1]:.3f}")

        # FPS+kNN: match GT seed count
        n_seeds = max(gt_n, 5)
        t0 = time.time()
        tok = tokenize_fps_knn(xyz, n_seeds=n_seeds)
        elapsed = time.time() - t0
        results["fps_knn_gt"]["time"].append(elapsed)
        results["fps_knn_gt"]["n_tokens"].append(n_tokens(tok))
        results["fps_knn_gt"]["ari"].append(adjusted_rand_index(gt_tokens, tok))
        print(f"  FPS+{n_seeds:3d}:    {elapsed*1000:.1f}ms, {n_tokens(tok):4d} tok, ARI={results['fps_knn_gt']['ari'][-1]:.3f}")

        # FPS+kNN: 50 seeds (中间粒度)
        t0 = time.time()
        tok = tokenize_fps_knn(xyz, n_seeds=50)
        elapsed = time.time() - t0
        results["fps_knn_50"]["time"].append(elapsed)
        results["fps_knn_50"]["n_tokens"].append(n_tokens(tok))
        results["fps_knn_50"]["ari"].append(adjusted_rand_index(gt_tokens, tok))
        print(f"  FPS+ 50:   {elapsed*1000:.1f}ms, {n_tokens(tok):4d} tok, ARI={results['fps_knn_50']['ari'][-1]:.3f}")

        # Superpoint (法向区域生长, k=15, 30°)
        t0 = time.time()
        tok = tokenize_superpoint(xyz, normal, k=15, angle_thresh_deg=30.0)
        elapsed = time.time() - t0
        results["superpoint"]["time"].append(elapsed)
        results["superpoint"]["n_tokens"].append(n_tokens(tok))
        results["superpoint"]["ari"].append(adjusted_rand_index(gt_tokens, tok))
        print(f"  SuperPt:   {elapsed*1000:.1f}ms, {n_tokens(tok):4d} tok, ARI={results['superpoint']['ari'][-1]:.3f}")

        # Superpoint-Lite: 更粗的参数 (k=8, 45°) → 更快更少 token
        t0 = time.time()
        tok = tokenize_superpoint(xyz, normal, k=8, angle_thresh_deg=45.0)
        elapsed = time.time() - t0
        results["superp_lite"]["time"].append(elapsed)
        results["superp_lite"]["n_tokens"].append(n_tokens(tok))
        results["superp_lite"]["ari"].append(adjusted_rand_index(gt_tokens, tok))
        print(f"  SuperLite: {elapsed*1000:.1f}ms, {n_tokens(tok):4d} tok, ARI={results['superp_lite']['ari'][-1]:.3f}")

    print("\n" + "=" * 60)
    print(f"{'Method':<12} {'Avg Time':>10} {'Avg #Tokens':>12} {'Avg ARI':>10}")
    print("-" * 60)
    for method in ["voxel_5cm", "voxel_8cm", "fps_knn_gt", "fps_knn_50", "superpoint", "superp_lite"]:
        r = results[method]
        avg_t = np.mean(r["time"]) * 1000
        avg_n = np.mean(r["n_tokens"])
        avg_ari = np.mean(r["ari"])
        print(f"{method:<12} {avg_t:>7.1f} ms  {avg_n:>9.1f}      {avg_ari:>7.3f}")
    print("=" * 60)
    print("ARI = Adjusted Rand Index vs PartSAM ground-truth tokens (越高越接近)")


if __name__ == "__main__":
    main()
