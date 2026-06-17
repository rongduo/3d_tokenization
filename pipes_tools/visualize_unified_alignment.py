#!/usr/bin/env python3
"""Render a grid of (rotated point cloud, mesh sample) overlays to visually
verify the unified rotation found by find_unified_rotation.py.

Reads rotation_search.json, picks N samples (one per category if possible),
produces a single PNG with two columns per row:
  col 0: pc rotated by R_unified only
  col 1: pc rotated by R_unified, then optimal per-object yaw

Plus a third column showing identity (no rotation) for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

# Reuse helpers from the search script.
from pipes_tools.find_unified_rotation import (  # type: ignore
    load_mesh_concat,
    load_points,
    normalize,
    sample_mesh_points,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_json", type=Path,
                   default=Path("/data5/jl/project/tokenizer_seg/align_unified/rotation_search.json"))
    p.add_argument("--train_root", type=Path,
                   default=Path("/data5/jl/project/training_data_3dcompat/trainingdata/trainingdata"))
    p.add_argument("--mesh_root", type=Path,
                   default=Path("/data3/jl/dataset/3DCoMPaT200/datasets--CoMPaT--3DCoMPaT200/"
                                "snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/Compat200/models"))
    p.add_argument("--n_panels", type=int, default=12)
    p.add_argument("--n_points", type=int, default=4096)
    p.add_argument("--out_png", type=Path,
                   default=Path("/data5/jl/project/tokenizer_seg/unified_rotation_panels.png"))
    return p.parse_args()


def yaw_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def scatter_two(ax, pcd: np.ndarray, mesh: np.ndarray, title: str) -> None:
    ax.scatter(pcd[:, 0], pcd[:, 1], s=0.3, c="red", alpha=0.45)
    ax.scatter(mesh[:, 0], mesh[:, 1], s=0.3, c="blue", alpha=0.45)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main() -> None:
    args = parse_args()
    data = json.loads(args.results_json.read_text())

    R_unified = np.asarray(
        data["rotations"][data["best_rotation_idx_by_mean"]]["matrix"], dtype=np.float32
    )
    yaw_angles = data.get("yaw_refinement", {}).get("best_yaw_angles_rad", [])
    per_sample = data["per_sample"]

    # Pick one per category if possible.
    seen = set()
    chosen = []
    for i, s in enumerate(per_sample):
        key = s["category"] if s["category"] != "unknown" else f"u{i}"
        if key not in seen:
            seen.add(key)
            chosen.append(i)
        if len(chosen) >= args.n_panels:
            break
    if len(chosen) < args.n_panels:
        for i in range(len(per_sample)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= args.n_panels:
                break

    fig, axes = plt.subplots(len(chosen), 3, figsize=(8.5, 2.6 * len(chosen)))
    rng = np.random.default_rng(42)

    headers = ["raw (no rot)", "+ unified rot", "+ unified + best yaw"]
    for col, h in enumerate(headers):
        axes[0, col].set_title(h + "\n" + axes[0, col].get_title(), fontsize=9)

    for row, idx in enumerate(chosen):
        s = per_sample[idx]
        mid = s["model_id"]
        cat = s["category"]
        ppt = args.train_root / f"coarse_b'{mid}'" / "points.pt"
        gltf = args.mesh_root / f"{mid}.gltf"
        if not ppt.exists() or not gltf.exists():
            for col in range(3):
                axes[row, col].set_visible(False)
            continue
        try:
            pcd = load_points(ppt)
            if pcd.shape[0] > args.n_points:
                idx_sub = torch.randperm(pcd.shape[0])[: args.n_points]
                pcd = pcd[idx_sub]
            pcd_n = normalize(pcd).numpy()
            mesh = load_mesh_concat(gltf)
            mpts = sample_mesh_points(mesh, args.n_points, rng)
            mpts_n = normalize(torch.from_numpy(mpts)).numpy()
        except Exception as exc:
            print(f"[warn] {mid}: {exc}")
            continue

        pcd_unified = (pcd_n @ R_unified.T).astype(np.float32)

        if idx < len(yaw_angles):
            theta = float(yaw_angles[idx])
        else:
            theta = 0.0
        Ryaw = yaw_matrix(theta)
        pcd_full = (pcd_unified @ Ryaw.T).astype(np.float32)

        title_left = f"{cat}/{mid}"
        title_mid = f"unified  cd={s['all_chamfer'][data['best_rotation_idx_by_mean']]:.4f}"
        title_right = f"yaw={np.rad2deg(theta):+.0f}deg"

        scatter_two(axes[row, 0], pcd_n, mpts_n, title_left)
        scatter_two(axes[row, 1], pcd_unified, mpts_n, title_mid)
        scatter_two(axes[row, 2], pcd_full, mpts_n, title_right)

    fig.suptitle(
        "Red = point cloud (training), Blue = mesh sample\n"
        f"Unified R (idx={data['best_rotation_idx_by_mean']}) = {R_unified.tolist()}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=120)
    print(f"[ok] saved {args.out_png}")


if __name__ == "__main__":
    main()
