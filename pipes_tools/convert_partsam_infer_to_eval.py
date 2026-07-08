#!/usr/bin/env python3
"""Map PartSAM infer outputs -> eval parts_partsam.pt (with graph cut).

Pipeline matches PartSAM/evaluation/postprocess_cpu.py used for training:
  1. Load GPU inference artifacts (sorted_masks, vertices, faces, point_to_face)
  2. Vote point masks -> per-face labels
  3. post_processing() with graph cut (default iou_threshold=0.65)
  4. Map face labels -> sampled points -> eval points.pt via NN

Eval rotation is unified for all categories:
    R = [[1,0,0],[0,0,1],[0,-1,0]];  pc_aligned = points @ R.T

Usage:
    PARTSAM_ROOT=/data5/jl/project/PartSAM python pipes_tools/convert_partsam_infer_to_eval.py \
        --infer_dir /path/to/infer_out/test_eval \
        --obj_list /path/to/test_obj_list.txt \
        --num_workers 16
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors

PARTSAM_ROOT = os.environ.get("PARTSAM_ROOT", "/data5/jl/project/PartSAM")
if PARTSAM_ROOT not in sys.path:
    sys.path.insert(0, PARTSAM_ROOT)
from utils.infer_utils import post_processing  # noqa: E402

EVAL_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def parse_uid(name: str) -> str | None:
    m = re.search(r"'([^']+)'", name)
    if m:
        return m.group(1)
    if name.startswith("coarse_b") or name.startswith("fine_b"):
        return name.split("b", 1)[-1].strip("_")
    return None


def rotate_points(points: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (points @ r.T).astype(np.float32)


def unit_normalize(pts: np.ndarray) -> np.ndarray:
    lo, hi = pts.min(0), pts.max(0)
    return (pts - (lo + hi) * 0.5) / (float((hi - lo).max()) + 1e-9)


def load_inference(infer_dir: Path, uid: str):
    d = infer_dir / uid
    return (
        np.load(d / "sampled_coords.npy"),
        np.load(d / "sorted_masks.npy"),
        np.load(d / "vertices.npy"),
        np.load(d / "faces.npy"),
        np.load(d / "point_to_face.npy"),
    )


def face_labels_from_masks(masks, vertices, faces, p2f, use_graph_cut, base_cfg):
    """Vote point masks to faces, run graph-cut post_processing, return face labels."""
    n_faces = len(faces)
    n_masks = len(masks)

    pt_labels = np.full(len(p2f), -1, dtype=np.int32)
    for i in range(n_masks):
        pt_labels[masks[i]] = i

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    votes = np.zeros((n_faces, n_masks), dtype=np.int32)
    np.add.at(votes, (p2f, pt_labels), 1)
    mvl = np.argmax(votes, axis=1)
    mvl[np.all(votes == 0, axis=1)] = -1

    vm = mvl != -1
    if not vm.all():
        ct = mesh.triangles_center
        nn = NearestNeighbors(n_neighbors=1).fit(ct[vm])
        mvl[~vm] = mvl[vm][nn.kneighbors(ct[~vm], return_distance=False).ravel()]

    eval_cfg = DictConfig({**base_cfg, "use_graph_cut": use_graph_cut})
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    mesh = post_processing(mvl.copy(), mesh, eval_cfg)

    fc = np.asarray(mesh.visual.face_colors)[:, :3]
    uniq, inv = np.unique(fc, axis=0, return_inverse=True)
    face_labels = inv.astype(np.int64).copy()
    black = np.where(np.all(uniq == 0, axis=1))[0]
    if len(black):
        face_labels[face_labels == black[0]] = -1
    return face_labels


def group_obj_dirs(obj_list_path: Path) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for line in obj_list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        uid = parse_uid(Path(line).name)
        if uid:
            groups[uid].append(line)
    return groups


def process_uid(args: Tuple) -> Tuple[str, str]:
    uid, obj_dirs, infer_dir_str, max_faces, eval_cfg, suffix, skip_existing = args
    infer_dir = Path(infer_dir_str)

    if not (infer_dir / uid / "sorted_masks.npy").exists():
        return uid, "missing_infer"

    try:
        sampled, masks, vertices, faces_np, p2f = load_inference(infer_dir, uid)
    except Exception as exc:
        return uid, f"load_failed:{exc}"

    n_faces = len(faces_np)
    use_gc = n_faces <= max_faces
    try:
        face_labels = face_labels_from_masks(
            masks, vertices, faces_np, p2f, use_gc, eval_cfg
        )
    except Exception as exc:
        return uid, f"gc_failed:{exc}"

    sampled_labels = face_labels[p2f]
    n_face_parts = len(set(face_labels[face_labels >= 0]))

    results = []
    for obj_dir_str in obj_dirs:
        obj_dir = Path(obj_dir_str)
        out_path = obj_dir / f"parts{suffix}.pt"
        if skip_existing and out_path.exists():
            results.append(f"{obj_dir.name}:skip")
            continue
        if not (obj_dir / "points.pt").exists():
            results.append(f"{obj_dir.name}:no_points")
            continue
        try:
            pts = torch.load(obj_dir / "points.pt", map_location="cpu", weights_only=True)
            if isinstance(pts, torch.Tensor):
                pts = pts.float().numpy()
            else:
                pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim == 3:
                pts = pts[0]

            pts_rot = rotate_points(pts, EVAL_ROTATION)
            src_n = unit_normalize(sampled)
            tgt_n = unit_normalize(pts_rot)
            nn = NearestNeighbors(n_neighbors=1).fit(src_n)
            nn_idx = nn.kneighbors(tgt_n, return_distance=False).ravel()
            train_labels = sampled_labels[nn_idx]

            torch.save(torch.from_numpy(train_labels.astype(np.int32)), out_path)
            n_parts = len(set(train_labels[train_labels >= 0]))
            assigned = int((train_labels >= 0).sum())
            results.append(f"{obj_dir.name}:{n_parts}p/{assigned}")
        except Exception as exc:
            results.append(f"{obj_dir.name}:fail({exc})")

    return uid, f"faces={n_faces} gc={use_gc} face_parts={n_face_parts} | " + " ".join(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_dir", type=Path, required=True)
    ap.add_argument("--obj_list", type=Path, required=True)
    ap.add_argument("--suffix", type=str, default="_partsam")
    ap.add_argument("--max_faces", type=int, default=50000,
                    help="skip graph_cut when mesh has more faces than this")
    ap.add_argument("--iou_threshold", type=float, default=0.65)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    eval_cfg = {
        "threshold_percentage_size": 0.01,
        "threshold_percentage_area": 0.01,
        "use_graph_cut": True,
        "iou_threshold": args.iou_threshold,
        "nms_threshold": 0.3,
    }

    uid_groups = group_obj_dirs(args.obj_list)
    uids = sorted(uid_groups.keys())
    if args.limit > 0:
        uids = uids[: args.limit]

    todo = [
        (uid, uid_groups[uid], str(args.infer_dir), args.max_faces, eval_cfg, args.suffix, args.skip_existing)
        for uid in uids
    ]

    ok = fail = 0
    t0 = time.time()
    with Pool(args.num_workers) as pool:
        for uid, msg in pool.imap_unordered(process_uid, todo, chunksize=4):
            if "fail" in msg.lower() or "failed" in msg.lower() or msg.startswith("missing") or msg.startswith("gc_failed"):
                fail += 1
                print(f"[FAIL] {uid}: {msg}")
            else:
                ok += 1
                if ok <= 5 or ok % 100 == 0:
                    print(f"[{ok+fail}/{len(todo)}] {uid}: {msg}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. uids ok={ok} fail={fail} total={len(todo)}")


if __name__ == "__main__":
    main()
