"""
为所有 test 对象生成 superpoint-based tokens，存为 parts_superpoint.pt
用法: python pipes_tools/generate_superpoint_tokens.py
"""
import torch
import numpy as np
import os, sys, time, glob

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

TEST_ROOT = "/data5/jl/project/training_data_3dcompat/test_3dcompat200/3dcompat200"


def tokenize_superpoint(xyz: np.ndarray, normal: np.ndarray,
                        k: int = 15, angle_thresh_deg: float = 30.0,
                        min_size: int = 3):
    N = xyz.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xyz)
    _, idx = nn.kneighbors(xyz)
    idx = idx[:, 1:]

    n_norm = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)
    ni = n_norm[:, None, :]
    nj = n_norm[idx]
    cos_angles = np.abs(np.sum(ni * nj, axis=2))
    valid_mask = cos_angles > np.cos(np.deg2rad(angle_thresh_deg))

    row_inds = np.repeat(np.arange(N), k)[valid_mask.ravel()]
    col_inds = idx.ravel()[valid_mask.ravel()]
    if len(row_inds) == 0:
        return np.zeros(N, dtype=np.int64)

    data = np.ones(len(row_inds), dtype=np.int8)
    adj_mat = csr_matrix((data, (row_inds, col_inds)), shape=(N, N))
    _, labels = connected_components(adj_mat, directed=False)

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


def main():
    files = sorted(glob.glob(f"{TEST_ROOT}/*/*/points.pt"))
    total = len(files)
    print(f"Total test objects: {total}")

    t0 = time.time()
    for i, pt_file in enumerate(files):
        obj_dir = os.path.dirname(pt_file)
        out_path = os.path.join(obj_dir, "parts_superpoint.pt")
        if os.path.exists(out_path):
            continue

        xyz = torch.load(pt_file, weights_only=True).numpy().astype(np.float32)
        normal = torch.load(os.path.join(obj_dir, "normals.pt"), weights_only=True).numpy().astype(np.float32)

        tokens = tokenize_superpoint(xyz, normal)
        torch.save(torch.from_numpy(tokens), out_path)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{total}  ({elapsed:.1f}s, {(i+1)/elapsed:.0f} obj/s)")

    elapsed = time.time() - t0
    print(f"Done. {total} objects in {elapsed:.1f}s ({total/elapsed:.0f} obj/s)")


if __name__ == "__main__":
    main()
