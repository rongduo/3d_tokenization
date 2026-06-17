#!/usr/bin/env python3
"""Benchmark mesh part operators against glTF fine-part GT (no point cloud / rotation).

Each glTF node name (e.g. airplane_door_01) is a semantic part mesh. This script
loads that GT directly from the mesh, runs operators such as mesh superpoint or
CoACD, and reports mixed-token / ARI / mean-IoU on sampled surface points.

Usage
-----
cd cosmo3d_other_dirs_excluding_main
/home/jl/anaconda3/envs/newpipelinefind3d/bin/python \\
  -m pipes_tools.benchmark_mesh_fine_parts \\
  --n_samples 20 --seed 0 --fast \\
  --out_json /data5/jl/project/tokenizer_seg/mesh_fine_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from sklearn.metrics import adjusted_rand_score

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipes_tools.assign_pointcloud_parts_by_mesh import load_mesh_fine_part_gt
from pipes_tools.benchmark_token_methods import (
    coacd_parts_to_mesh_superpoint,
    mesh_face_superpoint,
    mixed_token_stats,
    run_coacd_raw,
    transfer_face_labels_to_points,
)
from pipes_tools.benchmark_coacd_fine import parts_to_face_labels
from pipes_tools.compare_token_methods import adjusted_rand_index

DEFAULT_MESH_ROOT = Path(
    "/data3/jl/dataset/3DCoMPaT200/datasets--CoMPaT--3DCoMPaT200/"
    "snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/Compat200/models"
)


def mean_part_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = np.asarray(gt).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    ious: List[float] = []
    for g in np.unique(gt):
        g_mask = gt == g
        best = 0.0
        for p in np.unique(pred):
            inter = int(np.sum(g_mask & (pred == p)))
            if inter == 0:
                continue
            union = int(np.sum(g_mask | (pred == p)))
            best = max(best, inter / max(union, 1))
        ious.append(best)
    return float(np.mean(ious)) if ious else 0.0


def sample_eval_points(
    mesh: trimesh.Trimesh,
    face_gt: np.ndarray,
    n_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    face_idx = np.clip(face_idx.astype(np.int64), 0, len(face_gt) - 1)
    return pts.astype(np.float32), face_gt[face_idx]


def eval_method(
    gt: np.ndarray,
    pred_face_labels: np.ndarray,
    mesh_for_transfer: trimesh.Trimesh,
    eval_pts: np.ndarray,
) -> Dict[str, float]:
    pred = transfer_face_labels_to_points(mesh_for_transfer, pred_face_labels, eval_pts)
    stats = mixed_token_stats(gt, pred)
    stats["ari"] = float(adjusted_rand_index(gt, pred))
    stats["mean_part_iou"] = mean_part_iou(gt, pred)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Mesh-only fine-part operator benchmark")
    ap.add_argument("--mesh_root", type=Path, default=DEFAULT_MESH_ROOT)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--n_eval_points", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--merge_instances",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge airplane_door_01/02 into one fine part (default: True)",
    )
    ap.add_argument("--fast", action="store_true", help="Skip slow CoACD t=0.02")
    ap.add_argument(
        "--out_json",
        type=Path,
        default=Path("/data5/jl/project/tokenizer_seg/mesh_fine_benchmark.json"),
    )
    args = ap.parse_args()

    gltfs = sorted(args.mesh_root.glob("*.gltf"))
    random.seed(args.seed)
    sample = random.sample(gltfs, min(args.n_samples, len(gltfs)))

    method_names = ["mesh_sp30", "mesh_sp45", "mesh_sp60", "coacd_t010", "coacd002_sp45"]
    if not args.fast:
        method_names.insert(-1, "coacd_t002")

    agg: Dict[str, List[Dict[str, float]]] = {k: [] for k in method_names}
    per_object: List[Dict] = []
    skipped = 0

    coacd_kw = dict(
        resolution=2000,
        mcts_nodes=20,
        mcts_iterations=150,
        max_faces=-1,
        quiet=True,
    )
    if args.fast:
        coacd_kw.update(resolution=500, mcts_iterations=20, max_faces=5000)

    t0 = time.time()
    for i, gltf in enumerate(sample):
        mid = gltf.stem
        try:
            mesh, face_gt, part_names = load_mesh_fine_part_gt(
                gltf, merge_instances=args.merge_instances
            )
            eval_pts, gt = sample_eval_points(mesh, face_gt, args.n_eval_points, args.seed + i)
        except Exception as exc:
            skipped += 1
            print(f"[skip] {mid}: {exc}")
            continue

        obj_rec = {
            "model_id": mid,
            "n_faces": len(mesh.faces),
            "n_gt_parts": len(part_names),
            "gt_parts": part_names,
            "methods": {},
        }

        for angle, name in [(30.0, "mesh_sp30"), (45.0, "mesh_sp45"), (60.0, "mesh_sp60")]:
            fl = mesh_face_superpoint(mesh, angle_deg=angle)
            stats = eval_method(gt, fl, mesh, eval_pts)
            agg[name].append(stats)
            obj_rec["methods"][name] = stats

        parts_for_sp45: Optional[List] = None
        n_hulls_sp45 = 0

        for cname, threshold in [("coacd_t010", 0.10), ("coacd_t002", 0.02)]:
            if args.fast and cname == "coacd_t002":
                continue
            t1 = time.time()
            parts = run_coacd_raw(mesh, threshold=threshold, **coacd_kw)
            dt = time.time() - t1
            if parts is None:
                print(f"[warn] {mid} {cname}: CoACD failed")
                continue
            m2, fp, n_hulls = parts_to_face_labels(parts)
            stats = eval_method(gt, fp, m2, eval_pts)
            stats["n_hulls"] = float(n_hulls)
            stats["coacd_time_s"] = dt
            agg[cname].append(stats)
            obj_rec["methods"][cname] = stats
            parts_for_sp45 = parts
            n_hulls_sp45 = n_hulls

        if parts_for_sp45 is not None:
            m3, fp3 = coacd_parts_to_mesh_superpoint(parts_for_sp45, angle_deg=45.0)
            stats_sp = eval_method(gt, fp3, m3, eval_pts)
            stats_sp["n_hulls"] = float(n_hulls_sp45)
            agg["coacd002_sp45"].append(stats_sp)
            obj_rec["methods"]["coacd002_sp45"] = stats_sp

        per_object.append(obj_rec)
        sp45 = obj_rec["methods"].get("mesh_sp45", {})
        print(
            f"[{i+1}/{len(sample)}] {mid}  gt={len(part_names)} parts  "
            f"sp45 pt_mix={sp45.get('point_mixed_rate', float('nan')):.3f}  "
            f"ari={sp45.get('ari', float('nan')):.3f}  "
            f"elapsed={time.time()-t0:.0f}s"
        )

    summary = {}
    print("\n=== Mesh fine-part benchmark (no rotation) ===")
    print(
        f"{'method':<16} {'n':>3} {'pt_mix':>8} {'tok_mix':>8} "
        f"{'ari':>8} {'mIoU':>8} {'#tok':>7}"
    )
    for name in method_names:
        rows = agg[name]
        if not rows:
            print(f"{name:<16} {0:>3}  (no data)")
            continue
        summary[name] = {
            "count": len(rows),
            "point_mixed_rate": float(np.mean([r["point_mixed_rate"] for r in rows])),
            "token_mixed_rate": float(np.mean([r["token_mixed_rate"] for r in rows])),
            "mean_token_purity": float(np.mean([r["mean_token_purity"] for r in rows])),
            "mean_n_tokens": float(np.mean([r["n_tokens"] for r in rows])),
            "mean_ari": float(np.mean([r["ari"] for r in rows])),
            "mean_part_iou": float(np.mean([r["mean_part_iou"] for r in rows])),
        }
        if "coacd_time_s" in rows[0]:
            summary[name]["mean_coacd_time_s"] = float(np.mean([r["coacd_time_s"] for r in rows]))
        s = summary[name]
        print(
            f"{name:<16} {s['count']:>3} {s['point_mixed_rate']:>8.3f} "
            f"{s['token_mixed_rate']:>8.3f} {s['mean_ari']:>8.3f} "
            f"{s['mean_part_iou']:>8.3f} {s['mean_n_tokens']:>7.0f}"
        )

    out = {
        "summary": summary,
        "per_object": per_object,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "merge_instances": args.merge_instances,
        "skipped": skipped,
        "note": "GT from glTF node names; eval on sampled mesh surface points; no point-cloud rotation",
    }
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
