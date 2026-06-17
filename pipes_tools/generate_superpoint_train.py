"""为所有训练对象生成 k=15 superpoint tokens → parts_superpoint.pt"""
import torch, numpy as np, os, sys, time
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path: sys.path.insert(0, _src)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

TRAIN_TXT = "/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat/train.txt"


def superpoint(xyz, normal, k=15, angle_deg=30.0, min_size=3):
    N = xyz.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xyz)
    _, idx = nn.kneighbors(xyz)
    idx = idx[:, 1:]
    n_norm = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)
    cos = np.abs(np.sum(n_norm[:, None, :] * n_norm[idx], axis=2))
    valid = cos > np.cos(np.deg2rad(angle_deg))
    rows = np.repeat(np.arange(N), k)[valid.ravel()]
    cols = idx.ravel()[valid.ravel()]
    if len(rows) == 0:
        return np.zeros(N, dtype=np.int64)
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(N, N))
    _, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    small = np.where(sizes < min_size)[0]
    big = np.where(sizes >= min_size)[0]
    if len(small) and len(big):
        big_c = np.array([xyz[labels == b].mean(0) for b in big])
        for s in small:
            m = labels == s
            if m.sum():
                labels[m] = big[np.argmin(np.linalg.norm(big_c - xyz[m].mean(0), axis=1))]
    _, tokens = np.unique(labels, return_inverse=True)
    return tokens.astype(np.int64)


def main():
    with open(TRAIN_TXT) as f:
        dirs = [l.strip() for l in f if l.strip()]
    total = len(dirs)
    done = 0
    t0 = time.time()
    for i, d in enumerate(dirs):
        out = os.path.join(d, "parts_superpoint.pt")
        if os.path.exists(out):
            done += 1
            continue
        xyz = torch.load(os.path.join(d, "points.pt"), weights_only=True).numpy().astype(np.float32)
        normal = torch.load(os.path.join(d, "normals.pt"), weights_only=True).numpy().astype(np.float32)
        tok = superpoint(xyz, normal, k=15)
        torch.save(torch.from_numpy(tok), out)
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{total}  ({elapsed:.0f}s, {(i+1-done)/elapsed:.0f} obj/s)")
    elapsed = time.time() - t0
    print(f"Done. Generated {total-done}, skipped {done}. {elapsed:.0f}s ({(total-done)/elapsed:.0f} obj/s)")


if __name__ == "__main__":
    main()
