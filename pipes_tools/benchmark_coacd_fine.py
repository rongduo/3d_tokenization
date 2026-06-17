#!/usr/bin/env python3
"""Compare CoACD threshold=0.02 vs 0.10 on mixed-token quality."""
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

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipes_tools.benchmark_token_methods import (
    DEFAULT_MESH_ROOT,
    DEFAULT_TRAIN_TXT,
    R_UNIFIED,
    best_yaw,
    coacd_parts_to_mesh_superpoint,
    mixed_token_stats,
    mesh_face_superpoint,
    parse_model_id,
    run_coacd_raw,
    transfer_face_labels_to_points,
)
from pipes_tools.find_unified_rotation import load_mesh_concat

FOLDER_RE = re.compile(r"^(?:coarse|fine)_b'(?P<id>[^']+)'$")

COACD_DEFAULT = dict(
    resolution=2000,
    mcts_nodes=20,
    mcts_iterations=150,
    max_faces=-1,
    quiet=True,
)


def parts_to_face_labels(parts: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[trimesh.Trimesh, np.ndarray, int]:
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
    merged = trimesh.util.concatenate(meshes)
    face_part = np.concatenate(chunks, axis=0).astype(np.int64)
    return merged, face_part, len(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", type=Path, default=DEFAULT_TRAIN_TXT)
    ap.add_argument("--mesh_root", type=Path, default=DEFAULT_MESH_ROOT)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_json",
        type=Path,
        default=Path("/data5/jl/project/tokenizer_seg/coacd_fine_benchmark.json"),
    )
    args = ap.parse_args()

    with open(args.train_txt) as f:
        dirs = [l.strip() for l in f if l.strip() and "coarse_b" in l.strip()]
    random.seed(args.seed)
    sample = random.sample(dirs, min(args.n_samples, len(dirs)))

    methods = ["mesh_sp45", "coacd_t010", "coacd_t002", "coacd002_sp45"]
    agg: Dict[str, List[Dict]] = {k: [] for k in methods}
    per_object: List[Dict] = []

    t0 = time.time()
    for i, d in enumerate(sample):
        gt = torch.load(os.path.join(d, "parts.pt"), weights_only=True).numpy().reshape(-1)
        xyz = torch.load(os.path.join(d, "points.pt"), weights_only=True).numpy().astype(np.float32)
        if gt.shape[0] != xyz.shape[0] or xyz.shape[0] != 5000:
            continue

        mid = parse_model_id(os.path.basename(d))
        gltf = args.mesh_root / f"{mid}.gltf" if mid else None
        if not (mid and gltf and gltf.exists()):
            print(f"[skip] no mesh for {d}")
            continue

        mesh = load_mesh_concat(gltf)
        yaw = best_yaw(xyz, mesh, seed=args.seed)
        th = np.deg2rad(yaw)
        c, s = np.cos(th), np.sin(th)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
        xa = xyz @ (ry @ R_UNIFIED).T

        obj_rec = {"folder": os.path.basename(d), "model_id": mid, "orig_faces": len(mesh.faces), "methods": {}}

        fl = mesh_face_superpoint(mesh, angle_deg=45.0)
        tok = transfer_face_labels_to_points(mesh, fl, xa)
        stats = mixed_token_stats(gt, tok)
        agg["mesh_sp45"].append(stats)
        obj_rec["methods"]["mesh_sp45"] = {**stats, "n_hulls": stats["n_tokens"]}

        for name, threshold in [("coacd_t010", 0.10), ("coacd_t002", 0.02)]:
            t1 = time.time()
            parts = run_coacd_raw(mesh, threshold=threshold, **COACD_DEFAULT)
            dt = time.time() - t1
            if parts is None:
                print(f"[warn] {mid} {name}: CoACD failed")
                continue
            m2, fp, n_hulls = parts_to_face_labels(parts)
            tok = transfer_face_labels_to_points(m2, fp, xa)
            stats = mixed_token_stats(gt, tok)
            stats["n_hulls"] = float(n_hulls)
            stats["coacd_time_s"] = dt
            agg[name].append(stats)
            obj_rec["methods"][name] = stats

            if name == "coacd_t002":
                m3, fp3 = coacd_parts_to_mesh_superpoint(parts, angle_deg=45.0)
                tok_sp = transfer_face_labels_to_points(m3, fp3, xa)
                stats_sp = mixed_token_stats(gt, tok_sp)
                stats_sp["n_hulls"] = float(n_hulls)
                agg["coacd002_sp45"].append(stats_sp)
                obj_rec["methods"]["coacd002_sp45"] = stats_sp

        per_object.append(obj_rec)
        print(f"[{i+1}/{len(sample)}] {mid}  t010_hulls={obj_rec['methods'].get('coacd_t010',{}).get('n_hulls','?')}  "
              f"t002_hulls={obj_rec['methods'].get('coacd_t002',{}).get('n_hulls','?')}  "
              f"elapsed={time.time()-t0:.0f}s")

    summary = {}
    print("\n=== CoACD fine vs coarse (website default params, no mesh simplify) ===")
    print(f"{'method':<16} {'n':>3} {'hulls':>6} {'tok_mix':>8} {'pt_mix':>8} {'purity':>8} {'#tok':>7} {'mix_sz':>7}")
    for name in methods:
        rows = agg[name]
        if not rows:
            print(f"{name:<16} {0:>3}  (no data)")
            continue
        hull_key = "n_hulls" if "n_hulls" in rows[0] else "n_tokens"
        summary[name] = {
            "count": len(rows),
            "mean_n_hulls": float(np.mean([r.get("n_hulls", r["n_tokens"]) for r in rows])),
            "token_mixed_rate": float(np.mean([r["token_mixed_rate"] for r in rows])),
            "point_mixed_rate": float(np.mean([r["point_mixed_rate"] for r in rows])),
            "mean_token_purity": float(np.mean([r["mean_token_purity"] for r in rows])),
            "mean_n_tokens": float(np.mean([r["n_tokens"] for r in rows])),
            "mean_mixed_token_size": float(np.mean([r["mean_mixed_token_size"] for r in rows])),
        }
        if "coacd_time_s" in rows[0]:
            summary[name]["mean_coacd_time_s"] = float(np.mean([r["coacd_time_s"] for r in rows]))
        s = summary[name]
        print(
            f"{name:<16} {s['count']:>3} {s['mean_n_hulls']:>6.1f} "
            f"{s['token_mixed_rate']:>8.3f} {s['point_mixed_rate']:>8.3f} "
            f"{s['mean_token_purity']:>8.3f} {s['mean_n_tokens']:>7.0f} {s['mean_mixed_token_size']:>7.1f}"
        )

    out = {"summary": summary, "per_object": per_object, "seed": args.seed, "n_samples": args.n_samples}
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
