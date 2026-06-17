#!/usr/bin/env python3
"""Benchmark token methods including CoACD vs superpoint/voxel/FPS/mesh-SP."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipes_tools.compare_token_methods import (
    tokenize_fps_knn,
    tokenize_superpoint,
    tokenize_voxel,
)
from pipes_tools.find_unified_rotation import load_mesh_concat, normalize

R_UNIFIED = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], np.float32)
DEFAULT_MESH_ROOT = Path(
    "/data3/jl/dataset/3DCoMPaT200/datasets--CoMPaT--3DCoMPaT200/"
    "snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/Compat200/models"
)
DEFAULT_TRAIN_TXT = Path(
    "/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat/train.txt"
)
FOLDER_RE = re.compile(r"^(?:coarse|fine)_b'(?P<id>[^']+)'$")


def parse_model_id(folder_name: str) -> Optional[str]:
    m = FOLDER_RE.match(folder_name)
    return m.group("id") if m else None


def mixed_token_stats(gt: np.ndarray, tok: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt).reshape(-1)
    tok = np.asarray(tok).reshape(-1)
    n_pts = gt.shape[0]
    pure_sizes: List[int] = []
    mixed_sizes: List[int] = []
    mixed_tokens = 0
    pts_in_mixed = 0
    purities: List[float] = []

    for t in np.unique(tok):
        mask = tok == t
        _, cnt = np.unique(gt[mask], return_counts=True)
        purities.append(float(cnt.max() / mask.sum()))
        if len(cnt) >= 2:
            mixed_tokens += 1
            mixed_sizes.append(int(mask.sum()))
            pts_in_mixed += int(mask.sum())
        else:
            pure_sizes.append(int(mask.sum()))

    n_tok = len(np.unique(tok))
    return {
        "n_points": float(n_pts),
        "n_tokens": float(n_tok),
        "n_gt_parts": float(len(np.unique(gt))),
        "token_mixed_rate": mixed_tokens / max(n_tok, 1),
        "point_mixed_rate": pts_in_mixed / max(n_pts, 1),
        "mean_token_purity": float(np.mean(purities)) if purities else 1.0,
        "mean_pure_token_size": float(np.mean(pure_sizes)) if pure_sizes else 0.0,
        "mean_mixed_token_size": float(np.mean(mixed_sizes)) if mixed_sizes else 0.0,
    }


def best_yaw(xyz: np.ndarray, mesh: trimesh.Trimesh, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = min(512, xyz.shape[0])
    p = xyz[rng.choice(xyz.shape[0], n, replace=False)]
    mpts, _ = trimesh.sample.sample_surface(mesh, min(512, max(256, n)), seed=seed)
    p_n = normalize(torch.from_numpy(p.astype(np.float32))).numpy()
    m_n = normalize(torch.from_numpy(mpts.astype(np.float32))).numpy()
    best_cd, best_deg = 1e9, 0.0
    for deg in (0, 90, -90, 180):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
        pr = p_n @ ry.T
        cd = ((pr[:, None] - m_n[None, :]) ** 2).sum(-1).min(1).mean()
        cd += ((m_n[:, None] - pr[None, :]) ** 2).sum(-1).min(1).mean()
        if cd < best_cd:
            best_cd, best_deg = cd, float(deg)
    return best_deg


def prepare_mesh_for_coacd(mesh: trimesh.Trimesh, max_faces: int = 5000) -> trimesh.Trimesh:
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    if max_faces > 0 and len(m.faces) > max_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=max_faces)
        except Exception:
            idx = np.linspace(0, len(m.faces) - 1, max_faces, dtype=np.int64)
            m = trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces[idx], process=False)
    return m


def mesh_face_superpoint(mesh: trimesh.Trimesh, angle_deg: float = 45.0) -> np.ndarray:
    fn = np.asarray(mesh.face_normals, dtype=np.float32)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    adj = mesh.face_adjacency
    if len(adj) == 0:
        return np.zeros(len(fn), dtype=np.int64)
    i, j = adj[:, 0], adj[:, 1]
    cos = np.abs(np.sum(fn[i] * fn[j], axis=1))
    keep = cos > np.cos(np.deg2rad(angle_deg))
    rows = np.concatenate([i[keep], j[keep]])
    cols = np.concatenate([j[keep], i[keep]])
    mat = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(fn), len(fn)))
    _, labels = connected_components(mat, directed=False)
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64)


def transfer_face_labels_to_points(mesh: trimesh.Trimesh, face_labels: np.ndarray, pts: np.ndarray) -> np.ndarray:
    tri = np.asarray(mesh.triangles_center, dtype=np.float32)
    fi = NearestNeighbors(n_neighbors=1).fit(tri).kneighbors(pts)[1].ravel()
    fi = np.clip(fi, 0, len(face_labels) - 1)
    return face_labels[fi]


def run_coacd(
    mesh: trimesh.Trimesh,
    threshold: float = 0.1,
    max_faces: int = 5000,
    *,
    resolution: int = 500,
    mcts_iterations: int = 20,
    quiet: bool = True,
) -> Optional[Tuple[trimesh.Trimesh, np.ndarray]]:
    import coacd

    m = prepare_mesh_for_coacd(mesh, max_faces=max_faces)
    if len(m.faces) == 0:
        return None
    cm = coacd.Mesh(
        np.ascontiguousarray(m.vertices.astype(np.float64)),
        np.ascontiguousarray(m.faces.astype(np.int64)),
    )
    kwargs = dict(
        threshold=threshold,
        resolution=resolution,
        mcts_nodes=6,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=3,
        seed=0,
    )
    if quiet:
        with contextlib.redirect_stderr(io.StringIO()):
            parts = coacd.run_coacd(cm, **kwargs)
    else:
        parts = coacd.run_coacd(cm, **kwargs)
    if not parts:
        return None

    meshes: List[trimesh.Trimesh] = []
    chunks: List[np.ndarray] = []
    for pid, (vs, fs) in enumerate(parts):
        if len(fs) == 0:
            continue
        pm = trimesh.Trimesh(
            vertices=np.asarray(vs, dtype=np.float64),
            faces=np.asarray(fs, dtype=np.int64),
            process=False,
        )
        meshes.append(pm)
        chunks.append(np.full(len(pm.faces), pid, dtype=np.int32))
    if not meshes:
        return None
    merged = trimesh.util.concatenate(meshes)
    face_part = np.concatenate(chunks, axis=0)
    return merged, face_part.astype(np.int64)


def coacd_parts_to_mesh_superpoint(
    parts: List[Tuple[np.ndarray, np.ndarray]],
    angle_deg: float = 45.0,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    merged_list: List[trimesh.Trimesh] = []
    label_chunks: List[np.ndarray] = []
    offset = 0
    for vs, fs in parts:
        if len(fs) == 0:
            continue
        pm = trimesh.Trimesh(
            vertices=np.asarray(vs, dtype=np.float64),
            faces=np.asarray(fs, dtype=np.int64),
            process=False,
        )
        sp = mesh_face_superpoint(pm, angle_deg=angle_deg) + offset
        merged_list.append(pm)
        label_chunks.append(sp)
        offset = int(sp.max()) + 1
    merged = trimesh.util.concatenate(merged_list)
    face_labels = np.concatenate(label_chunks, axis=0)
    return merged, face_labels.astype(np.int64)


def run_coacd_raw(
    mesh: trimesh.Trimesh,
    threshold: float = 0.1,
    max_faces: int = 5000,
    *,
    resolution: int = 500,
    mcts_nodes: int = 6,
    mcts_iterations: int = 20,
    quiet: bool = True,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    import coacd

    m = prepare_mesh_for_coacd(mesh, max_faces=max_faces)
    if len(m.faces) == 0:
        return None
    cm = coacd.Mesh(
        np.ascontiguousarray(m.vertices.astype(np.float64)),
        np.ascontiguousarray(m.faces.astype(np.int64)),
    )
    kwargs = dict(
        threshold=threshold,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=3,
        seed=0,
    )
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parts = coacd.run_coacd(cm, **kwargs)
    else:
        parts = coacd.run_coacd(cm, **kwargs)
    return parts if parts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", type=Path, default=DEFAULT_TRAIN_TXT)
    ap.add_argument("--mesh_root", type=Path, default=DEFAULT_MESH_ROOT)
    ap.add_argument("--n_samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_coarse", action="store_true")
    ap.add_argument("--max_coacd_faces", type=int, default=5000)
    ap.add_argument("--fast", action="store_true", help="Fewer CoACD iterations / skip t=0.05")
    ap.add_argument("--out_json", type=Path, default=Path("/data5/jl/project/tokenizer_seg/token_benchmark.json"))
    args = ap.parse_args()

    coacd_resolution = 300 if args.fast else 500
    coacd_iters = 10 if args.fast else 20
    max_faces = min(args.max_coacd_faces, 3000 if args.fast else args.max_coacd_faces)

    with open(args.train_txt) as f:
        dirs = [l.strip() for l in f if l.strip()]
    if args.only_coarse:
        dirs = [d for d in dirs if "coarse_b" in d]

    random.seed(args.seed)
    sample = random.sample(dirs, min(args.n_samples, len(dirs)))

    method_names = [
        "pc_sp_k15", "pc_sp_k8", "pc_fps50", "pc_fps100", "pc_voxel5", "pc_voxel8",
        "mesh_sp45", "coacd_t010", "coacd_t005", "coacd010_sp45",
    ]
    agg: Dict[str, List[Dict[str, float]]] = {k: [] for k in method_names}
    skipped: Dict[str, int] = {k: 0 for k in method_names}

    t0 = time.time()
    for i, d in enumerate(sample):
        gt_path = os.path.join(d, "parts.pt")
        if not os.path.exists(gt_path):
            continue
        gt = torch.load(gt_path, weights_only=True).numpy().reshape(-1)
        xyz = torch.load(os.path.join(d, "points.pt"), weights_only=True).numpy().astype(np.float32)
        normal = torch.load(os.path.join(d, "normals.pt"), weights_only=True).numpy().astype(np.float32)
        if gt.shape[0] != xyz.shape[0]:
            continue
        if args.only_coarse and xyz.shape[0] != 5000:
            continue

        pc_methods = {
            "pc_sp_k15": lambda: tokenize_superpoint(xyz, normal, k=15, angle_thresh_deg=30.0),
            "pc_sp_k8": lambda: tokenize_superpoint(xyz, normal, k=8, angle_thresh_deg=45.0),
            "pc_fps50": lambda: tokenize_fps_knn(xyz, n_seeds=50),
            "pc_fps100": lambda: tokenize_fps_knn(xyz, n_seeds=100),
            "pc_voxel5": lambda: tokenize_voxel(xyz, voxel_size=0.05),
            "pc_voxel8": lambda: tokenize_voxel(xyz, voxel_size=0.08),
        }
        for name, fn in pc_methods.items():
            try:
                agg[name].append(mixed_token_stats(gt, fn()))
            except Exception:
                skipped[name] += 1

        mid = parse_model_id(os.path.basename(d))
        gltf = args.mesh_root / f"{mid}.gltf" if mid else None
        if not (mid and gltf and gltf.exists()):
            for k in ["mesh_sp45", "coacd_t010", "coacd_t005", "coacd010_sp45"]:
                skipped[k] += 1
            continue

        try:
            mesh = load_mesh_concat(gltf)
            yaw = best_yaw(xyz, mesh, seed=args.seed)
            th = np.deg2rad(yaw)
            c, s = np.cos(th), np.sin(th)
            ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
            xa = xyz @ (ry @ R_UNIFIED).T

            fl = mesh_face_superpoint(mesh, angle_deg=45.0)
            agg["mesh_sp45"].append(mixed_token_stats(gt, transfer_face_labels_to_points(mesh, fl, xa)))

            coacd_kw = dict(
                max_faces=max_faces,
                resolution=coacd_resolution,
                mcts_iterations=coacd_iters,
                quiet=True,
            )
            parts010 = run_coacd_raw(mesh, threshold=0.10, **coacd_kw)
            if parts010 is None:
                skipped["coacd_t010"] += 1
                skipped["coacd010_sp45"] += 1
            else:
                meshes: List[trimesh.Trimesh] = []
                chunks: List[np.ndarray] = []
                for pid, (vs, fs) in enumerate(parts010):
                    if len(fs) == 0:
                        continue
                    pm = trimesh.Trimesh(
                        vertices=np.asarray(vs, dtype=np.float64),
                        faces=np.asarray(fs, dtype=np.int64),
                        process=False,
                    )
                    meshes.append(pm)
                    chunks.append(np.full(len(pm.faces), pid, dtype=np.int32))
                m2 = trimesh.util.concatenate(meshes)
                fp = np.concatenate(chunks, axis=0).astype(np.int64)
                agg["coacd_t010"].append(mixed_token_stats(gt, transfer_face_labels_to_points(m2, fp, xa)))
                m3, fp3 = coacd_parts_to_mesh_superpoint(parts010, angle_deg=45.0)
                agg["coacd010_sp45"].append(mixed_token_stats(gt, transfer_face_labels_to_points(m3, fp3, xa)))

            if not args.fast:
                parts005 = run_coacd_raw(mesh, threshold=0.05, **coacd_kw)
                if parts005 is None:
                    skipped["coacd_t005"] += 1
                else:
                    meshes5: List[trimesh.Trimesh] = []
                    chunks5: List[np.ndarray] = []
                    for pid, (vs, fs) in enumerate(parts005):
                        if len(fs) == 0:
                            continue
                        pm = trimesh.Trimesh(
                            vertices=np.asarray(vs, dtype=np.float64),
                            faces=np.asarray(fs, dtype=np.int64),
                            process=False,
                        )
                        meshes5.append(pm)
                        chunks5.append(np.full(len(pm.faces), pid, dtype=np.int32))
                    m5 = trimesh.util.concatenate(meshes5)
                    fp5 = np.concatenate(chunks5, axis=0).astype(np.int64)
                    agg["coacd_t005"].append(mixed_token_stats(gt, transfer_face_labels_to_points(m5, fp5, xa)))
            else:
                skipped["coacd_t005"] += 1
        except Exception as exc:
            for k in ["mesh_sp45", "coacd_t010", "coacd_t005", "coacd010_sp45"]:
                skipped[k] += 1
            print(f"[warn] {os.path.basename(d)}: {exc}")

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(sample)}] elapsed={time.time()-t0:.1f}s")

    summary = {}
    print("\n=== Mixed-token benchmark ===")
    print(f"{'method':<16} {'n':>4} {'tok_mix':>8} {'pt_mix':>8} {'purity':>8} {'#tok':>8} {'mix_sz':>8}")
    for name in method_names:
        rows = agg[name]
        if not rows:
            print(f"{name:<16} {0:>4}  skipped={skipped[name]}")
            continue
        summary[name] = {
            "count": len(rows),
            "skipped": skipped[name],
            "token_mixed_rate": float(np.mean([r["token_mixed_rate"] for r in rows])),
            "point_mixed_rate": float(np.mean([r["point_mixed_rate"] for r in rows])),
            "mean_token_purity": float(np.mean([r["mean_token_purity"] for r in rows])),
            "mean_n_tokens": float(np.mean([r["n_tokens"] for r in rows])),
            "mean_mixed_token_size": float(np.mean([r["mean_mixed_token_size"] for r in rows])),
        }
        s = summary[name]
        print(
            f"{name:<16} {s['count']:>4} {s['token_mixed_rate']:>8.3f} {s['point_mixed_rate']:>8.3f} "
            f"{s['mean_token_purity']:>8.3f} {s['mean_n_tokens']:>8.0f} {s['mean_mixed_token_size']:>8.1f}"
        )

    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
