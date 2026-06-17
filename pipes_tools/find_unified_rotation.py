#!/usr/bin/env python3
"""Find a single (unified) axis-aligned rotation that aligns the training point clouds
in /data5/jl/project/training_data_3dcompat/trainingdata/trainingdata
with the GLTF meshes in
/data3/jl/dataset/3DCoMPaT200/.../Compat200/models.

Approach
--------
For N randomly sampled objects:
  1. Load points.pt -> Pcd (centered, scale-normalized).
  2. Load the matching .gltf, sample its surface -> Mesh (centered, scale-normalized).
  3. For each of the 24 proper axis-aligned rotations R in SO(3),
     compute Chamfer(Pcd @ R.T, Mesh) on GPU.
We then aggregate per-object best rotation (mode) and total Chamfer per rotation
(sum), so we can tell whether a single rotation truly explains the dataset.

Usage
-----
python -m pipes_tools.find_unified_rotation \
    --train_root /data5/jl/project/training_data_3dcompat/trainingdata/trainingdata \
    --mesh_root  /data3/jl/dataset/3DCoMPaT200/datasets--CoMPaT--3DCoMPaT200/snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/Compat200/models \
    --n_samples 200 --n_points 4096 --out_dir ./align_unified
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh


DEFAULT_TRAIN_ROOT = Path(
    "/data5/jl/project/training_data_3dcompat/trainingdata/trainingdata"
)
DEFAULT_MESH_ROOT = Path(
    "/data3/jl/dataset/3DCoMPaT200/datasets--CoMPaT--3DCoMPaT200/"
    "snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/Compat200/models"
)
DEFAULT_TEST_ROOT = Path(
    "/data5/jl/project/training_data_3dcompat/test_3dcompat200/3dcompat200"
)


# ---------- IO helpers ----------------------------------------------------- #

_FOLDER_RE = re.compile(r"^coarse_b'(?P<id>[^']+)'$")


def parse_model_id(folder_name: str) -> Optional[str]:
    """Folder names look like coarse_b'00_001' -> '00_001'."""
    m = _FOLDER_RE.match(folder_name)
    return m.group("id") if m else None


def load_points(points_path: Path) -> torch.Tensor:
    data = torch.load(points_path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        pts = data
    elif isinstance(data, np.ndarray):
        pts = torch.from_numpy(data)
    elif isinstance(data, dict):
        pts = None
        for key in ("points", "xyz", "pcd", "verts", "vertices"):
            if key in data:
                v = data[key]
                pts = v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
                break
        if pts is None:
            for v in data.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[-1] == 3:
                    pts = v
                    break
    else:
        raise ValueError(f"Unsupported points.pt format: {type(data)}")
    pts = pts.float()
    if pts.ndim == 3:
        pts = pts[0]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"bad points shape {tuple(pts.shape)} in {points_path}")
    return pts.contiguous()


def load_mesh_concat(mesh_path: Path) -> trimesh.Trimesh:
    """Load .gltf preserving scene-graph transforms, return a single Trimesh."""
    scene = trimesh.load(str(mesh_path), force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise ValueError(f"unsupported / empty mesh: {mesh_path}")

    parts: List[trimesh.Trimesh] = []
    # Walk the scene graph so per-node transforms are applied.
    for node_name in scene.graph.nodes_geometry:
        tf, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue
        m = geom.copy()
        m.apply_transform(tf)
        parts.append(m)
    if not parts:
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                parts.append(geom.copy())
    if not parts:
        raise ValueError(f"no Trimesh parts in: {mesh_path}")
    return trimesh.util.concatenate(parts)


def sample_mesh_points(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    if mesh.area <= 0:
        # Degenerate mesh – fall back to vertices.
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if verts.shape[0] >= n:
            idx = rng.choice(verts.shape[0], size=n, replace=False)
            return verts[idx]
        idx = rng.choice(verts.shape[0], size=n, replace=True)
        return verts[idx]
    seed = int(rng.integers(0, 2**31 - 1))
    pts, _ = trimesh.sample.sample_surface(mesh, count=n, seed=seed)
    return np.asarray(pts, dtype=np.float32)


def normalize(pts: torch.Tensor) -> torch.Tensor:
    """Center bbox at origin and scale so max axis extent = 1."""
    lo = pts.min(dim=0).values
    hi = pts.max(dim=0).values
    center = 0.5 * (lo + hi)
    pts = pts - center
    extent = (hi - lo).max().clamp(min=1e-6)
    return pts / extent


# ---------- Rotation enumeration ------------------------------------------ #

def enumerate_24_rotations() -> List[Tuple[str, np.ndarray]]:
    """All 24 proper rotations of the cube (signed axis permutations with det=+1)."""
    out: List[Tuple[str, np.ndarray]] = []
    eye = np.eye(3, dtype=np.float32)
    for perm in itertools.permutations(range(3)):
        base = eye[list(perm)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            r = base.copy()
            for i in range(3):
                r[i] *= signs[i]
            if np.linalg.det(r) > 0.99:
                tag = f"perm{perm}_signs{tuple(int(s) for s in signs)}"
                out.append((tag, r))
    assert len(out) == 24, len(out)
    return out


def rotation_pretty_name(r: np.ndarray) -> str:
    """Try to label R as identity / 90° rotation about X/Y/Z when applicable."""
    eye = np.eye(3, dtype=np.float32)
    if np.allclose(r, eye, atol=1e-3):
        return "identity"
    axis_names = ["X", "Y", "Z"]
    for axis in range(3):
        for deg, sign in ((90, +1), (-90, -1), (180, +1)):
            theta = np.deg2rad(deg) * sign if abs(deg) == 90 else np.deg2rad(180)
            c, s = np.cos(theta), np.sin(theta)
            r_axis = eye.copy()
            if axis == 0:
                r_axis = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
            elif axis == 1:
                r_axis = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            else:
                r_axis = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            if np.allclose(r, r_axis, atol=1e-3):
                lbl = f"{axis_names[axis]}{'+' if deg > 0 else ''}{deg if abs(deg)==90 else 180}"
                return f"rot_{lbl}"
    return "compound"


# ---------- Chamfer (batched, GPU) ---------------------------------------- #

def chamfer_batch(rotated_pcs: torch.Tensor, mesh_pts: torch.Tensor, chunk: int = 4) -> torch.Tensor:
    """rotated_pcs: [R, N, 3], mesh_pts: [N, 3] -> chamfer per rotation, [R].

    Chunks over R to bound memory: a [chunk, N, N] cdist per step.
    """
    r_total = rotated_pcs.shape[0]
    out = torch.empty(r_total, dtype=rotated_pcs.dtype, device=rotated_pcs.device)
    mesh_b = mesh_pts.unsqueeze(0)
    for s in range(0, r_total, chunk):
        e = min(s + chunk, r_total)
        sub = rotated_pcs[s:e]
        dmat = torch.cdist(sub, mesh_b.expand(sub.shape[0], -1, -1), p=2)  # [k, N, N]
        fwd = dmat.min(dim=2).values.pow(2).mean(dim=1)
        bwd = dmat.min(dim=1).values.pow(2).mean(dim=1)
        out[s:e] = fwd + bwd
    return out


# ---------- Main ---------------------------------------------------------- #

def build_id_to_category(test_root: Path) -> Dict[str, str]:
    """Map model_id -> category.

    Tries an exact lookup (rare since train/test ids are disjoint) AND a 2-char
    prefix lookup (model ids are like 'XX_YYY' where XX is the category prefix
    in 3DCoMPaT200).
    """
    out: Dict[str, str] = {}
    prefix_votes: Dict[str, Dict[str, int]] = {}
    if not test_root.exists():
        return out
    for cat_dir in test_root.iterdir():
        if not cat_dir.is_dir():
            continue
        for sample_dir in cat_dir.iterdir():
            mid = parse_model_id(sample_dir.name)
            if mid is None:
                continue
            out[mid] = cat_dir.name
            prefix = mid.split("_", 1)[0]
            prefix_votes.setdefault(prefix, {}).setdefault(cat_dir.name, 0)
            prefix_votes[prefix][cat_dir.name] += 1

    # Promote dominant prefix votes to a prefix-level mapping.
    prefix_to_cat: Dict[str, str] = {}
    for prefix, votes in prefix_votes.items():
        prefix_to_cat[prefix] = max(votes.items(), key=lambda kv: kv[1])[0]

    # Stash the prefix map under the special key so callers can use it.
    out["__PREFIX__"] = json.dumps(prefix_to_cat)  # type: ignore[assignment]
    return out


def lookup_category(id2cat: Dict[str, str], mid: str) -> str:
    if mid in id2cat and not mid.startswith("__"):
        return id2cat[mid]
    prefix_map_json = id2cat.get("__PREFIX__", "{}")
    try:
        prefix_map = json.loads(prefix_map_json)
    except Exception:
        prefix_map = {}
    return prefix_map.get(mid.split("_", 1)[0], "unknown")


def collect_train_samples(train_root: Path, mesh_root: Path) -> List[Tuple[str, Path, Path]]:
    """Return list of (model_id, points.pt, mesh.gltf) for samples that have both."""
    out: List[Tuple[str, Path, Path]] = []
    for d in train_root.iterdir():
        if not d.is_dir():
            continue
        mid = parse_model_id(d.name)
        if mid is None:
            continue
        ppt = d / "points.pt"
        gltf = mesh_root / f"{mid}.gltf"
        if ppt.exists() and gltf.exists():
            out.append((mid, ppt, gltf))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", type=Path, default=DEFAULT_TRAIN_ROOT)
    p.add_argument("--mesh_root", type=Path, default=DEFAULT_MESH_ROOT)
    p.add_argument("--test_root", type=Path, default=DEFAULT_TEST_ROOT,
                   help="Used only for model_id -> category lookup.")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_points", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chamfer_chunk", type=int, default=4,
                   help="Number of rotations to evaluate at once on GPU (memory knob).")
    p.add_argument("--out_dir", type=Path, default=Path("./align_unified"))
    p.add_argument("--export_top_k_visuals", type=int, default=5,
                   help="Export ply visualizations for k random samples after applying the best rotation.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[info] device={device}")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    id2cat = build_id_to_category(args.test_root)
    print(f"[info] id2cat from test set: {len(id2cat)} entries")

    all_pairs = collect_train_samples(args.train_root, args.mesh_root)
    print(f"[info] training samples with mesh available: {len(all_pairs)}")
    random.shuffle(all_pairs)
    pairs = all_pairs[: args.n_samples]
    print(f"[info] using {len(pairs)} samples")

    rotations = enumerate_24_rotations()
    rot_mats = torch.stack(
        [torch.from_numpy(r).float() for _, r in rotations], dim=0
    ).to(device)  # [24, 3, 3]

    per_sample: List[Dict] = []
    chamfer_sum = torch.zeros(24, dtype=torch.float64, device=device)
    chamfer_count = 0
    best_rot_count = np.zeros(24, dtype=np.int64)

    t0 = time.time()
    for i, (mid, ppt, gltf) in enumerate(pairs):
        try:
            pcd = load_points(ppt)  # [N, 3]
            if pcd.shape[0] > args.n_points:
                idx = torch.randperm(pcd.shape[0])[: args.n_points]
                pcd = pcd[idx]
            elif pcd.shape[0] < args.n_points:
                # repeat-pad so cdist batch shapes match.
                k = (args.n_points + pcd.shape[0] - 1) // pcd.shape[0]
                pcd = pcd.repeat(k, 1)[: args.n_points]

            mesh = load_mesh_concat(gltf)
            mpts = sample_mesh_points(mesh, args.n_points, rng)
        except Exception as exc:
            print(f"[warn] {mid}: {exc}")
            continue

        pcd_n = normalize(pcd).to(device)
        mpts_n = normalize(torch.from_numpy(mpts)).to(device)

        # rotated_pcs: [24, N, 3]  =  pcd @ R.T  for each R
        rotated = torch.einsum("rij,nj->rni", rot_mats, pcd_n)
        cd = chamfer_batch(rotated, mpts_n, chunk=args.chamfer_chunk)  # [24]

        cd_cpu = cd.detach().cpu().numpy()
        best_idx = int(np.argmin(cd_cpu))
        best_rot_count[best_idx] += 1
        chamfer_sum += cd.double()
        chamfer_count += 1

        per_sample.append({
            "model_id": mid,
            "category": lookup_category(id2cat, mid),
            "best_rot_idx": best_idx,
            "best_rot_name": rotations[best_idx][0],
            "best_chamfer": float(cd_cpu[best_idx]),
            "identity_chamfer": float(cd_cpu[0]),
            "all_chamfer": cd_cpu.tolist(),
        })

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(pairs)}] elapsed={time.time()-t0:.1f}s  best={rotations[best_idx][0]}")

    # ---- Aggregation ----
    chamfer_mean = (chamfer_sum / max(chamfer_count, 1)).cpu().numpy()
    rank = np.argsort(chamfer_mean)
    print("\n=== Aggregate (mean chamfer over samples) - lower is better ===")
    print(f"{'rank':>4} {'idx':>3} {'pretty':>10} {'mean_cd':>12} {'pretty_R':>20} {'best_count':>10}")
    for k, idx in enumerate(rank[:24]):
        name, r = rotations[idx]
        print(
            f"{k:>4} {idx:>3} {rotation_pretty_name(r):>10} "
            f"{chamfer_mean[idx]:>12.6f} {name:>30} {best_rot_count[idx]:>10}"
        )

    print("\n=== Per-sample best rotation histogram (top 5) ===")
    hist_order = np.argsort(-best_rot_count)
    for idx in hist_order[:5]:
        name, r = rotations[idx]
        pct = 100 * best_rot_count[idx] / max(chamfer_count, 1)
        print(f"  idx={idx:2d}  {rotation_pretty_name(r):>10}  count={best_rot_count[idx]:4d} ({pct:5.1f}%)  R={name}")

    # Per-category breakdown.
    cats: Dict[str, Dict[int, int]] = {}
    for s in per_sample:
        cat = s["category"]
        cats.setdefault(cat, {})
        cats[cat][s["best_rot_idx"]] = cats[cat].get(s["best_rot_idx"], 0) + 1
    print("\n=== Per-category dominant rotation (>=1 sample) ===")
    for cat, counts in sorted(cats.items(), key=lambda kv: -sum(kv[1].values()))[:30]:
        n = sum(counts.values())
        top_idx = max(counts.items(), key=lambda kv: kv[1])
        name, r = rotations[top_idx[0]]
        print(f"  {cat:>22s}  n={n:3d}  top={rotation_pretty_name(r):>10} ({top_idx[1]}/{n})")

    # ---- Continuous yaw refinement around the up-axis ----
    # The point of this analysis: if the dataset shares ONE up-axis swap and
    # each object only differs by a free yaw, then refining yaw around the
    # up-axis (after the unified discrete rotation) should drop the residual
    # chamfer to ~0 for every object.
    print("\n=== Continuous yaw refinement around mesh-Y after unified rotation ===")
    n_yaw = 360
    yaw_search_grid = np.linspace(-np.pi, np.pi, n_yaw, endpoint=False, dtype=np.float32)
    yaw_search_torch = torch.from_numpy(yaw_search_grid).to(device)
    cos_y = torch.cos(yaw_search_torch)
    sin_y = torch.sin(yaw_search_torch)
    yaw_mats = torch.zeros((yaw_search_torch.shape[0], 3, 3), device=device)
    yaw_mats[:, 0, 0] = cos_y
    yaw_mats[:, 0, 2] = sin_y
    yaw_mats[:, 1, 1] = 1.0
    yaw_mats[:, 2, 0] = -sin_y
    yaw_mats[:, 2, 2] = cos_y

    best_idx_global = int(rank[0])
    best_R = rotations[best_idx_global][1]
    R0_t = torch.from_numpy(best_R).float().to(device)
    yaw_residuals: List[float] = []
    yaw_best_angles: List[float] = []
    yaw_chunk = max(1, args.chamfer_chunk * 2)  # cdist alone, no rotation broadcasting overhead
    for s, (mid, ppt, gltf) in zip(per_sample, pairs[: len(per_sample)]):
        try:
            pcd = load_points(ppt)
            if pcd.shape[0] > args.n_points:
                idx = torch.randperm(pcd.shape[0])[: args.n_points]
                pcd = pcd[idx]
            pcd_n = normalize(pcd).to(device)
            mesh = load_mesh_concat(gltf)
            mpts = sample_mesh_points(mesh, args.n_points, rng)
            mpts_n = normalize(torch.from_numpy(mpts)).to(device)
            base = pcd_n @ R0_t.T
            cd_full = torch.empty(n_yaw, dtype=torch.float32, device=device)
            for ys in range(0, n_yaw, yaw_chunk):
                ye = min(ys + yaw_chunk, n_yaw)
                rot_sub = yaw_mats[ys:ye]
                rotated_sub = torch.einsum("rij,nj->rni", rot_sub, base)
                cd_full[ys:ye] = chamfer_batch(rotated_sub, mpts_n, chunk=args.chamfer_chunk)
                del rotated_sub
            yaw_residuals.append(float(cd_full.min().item()))
            yaw_best_angles.append(float(yaw_search_grid[int(cd_full.argmin().item())]))
            del cd_full, base
        except Exception as exc:
            print(f"[warn][yaw] {mid}: {exc}")
            torch.cuda.empty_cache() if device.type == "cuda" else None

    if yaw_residuals:
        print(f"  samples evaluated: {len(yaw_residuals)}")
        print(f"  unified discrete rot only - mean chamfer: {chamfer_mean[best_idx_global]:.6f}")
        print(f"  + continuous yaw refinement - mean chamfer: {np.mean(yaw_residuals):.6f}")
        print(f"  + continuous yaw refinement - median chamfer: {np.median(yaw_residuals):.6f}")
        print(f"  + continuous yaw refinement - 95th pct chamfer: {np.percentile(yaw_residuals, 95):.6f}")

        angles_deg = np.array(yaw_best_angles) * 180.0 / np.pi
        # snap to nearest 90 to see if yaw clusters around quadrants
        snapped = np.round(angles_deg / 90.0) * 90.0
        residual = np.minimum(np.abs(angles_deg - snapped),
                              np.abs(np.abs(angles_deg - snapped) - 360.0))
        print(f"  per-object yaw distance from nearest 90 multiple: "
              f"mean={residual.mean():.2f}deg  median={np.median(residual):.2f}deg  "
              f"max={residual.max():.2f}deg")
        bins = [-180, -135, -45, 45, 135, 180]
        labels = ["180", "-90", "0", "90", "180"]
        counts = np.histogram(angles_deg, bins=bins)[0]
        # Wrap the two 180 bins together for display.
        c180 = counts[0] + counts[-1]
        print("  yaw quadrant histogram:  "
              f"-90:{counts[1]:3d}  0:{counts[2]:3d}  90:{counts[3]:3d}  180:{c180:3d}")

    # Save full results.
    summary = {
        "n_samples_used": chamfer_count,
        "n_points": args.n_points,
        "rotations": [
            {
                "idx": i,
                "name": name,
                "pretty": rotation_pretty_name(r),
                "matrix": r.tolist(),
                "mean_chamfer": float(chamfer_mean[i]),
                "best_count": int(best_rot_count[i]),
            }
            for i, (name, r) in enumerate(rotations)
        ],
        "ranking_by_mean_chamfer": [int(i) for i in rank],
        "best_rotation_idx_by_mean": int(rank[0]),
        "best_rotation_idx_by_majority": int(np.argmax(best_rot_count)),
        "per_category_counts": {k: {str(kk): vv for kk, vv in v.items()} for k, v in cats.items()},
        "per_sample": per_sample,
        "yaw_refinement": {
            "samples_evaluated": len(yaw_residuals),
            "unified_only_mean_chamfer": float(chamfer_mean[int(rank[0])]),
            "yaw_refined_mean_chamfer": float(np.mean(yaw_residuals)) if yaw_residuals else None,
            "yaw_refined_median_chamfer": float(np.median(yaw_residuals)) if yaw_residuals else None,
            "yaw_refined_p95_chamfer": float(np.percentile(yaw_residuals, 95)) if yaw_residuals else None,
            "best_yaw_angles_rad": yaw_best_angles,
        },
    }
    out_json = args.out_dir / "rotation_search.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] full results -> {out_json}")

    # ---- Visualize top-k samples under the unified best rotation ----
    if args.export_top_k_visuals > 0 and chamfer_count > 0:
        vis_dir = args.out_dir / "visuals"
        vis_dir.mkdir(parents=True, exist_ok=True)
        best_idx_global = int(rank[0])
        best_R = rotations[best_idx_global][1]
        print(f"\n[info] exporting visuals for global best rot idx={best_idx_global} "
              f"({rotation_pretty_name(best_R)}) -> {vis_dir}")
        # Prefer one sample per category; fall back to first-K when categories collapse.
        seen_cats: set = set()
        export_pairs: List[Tuple[str, Path, Path]] = []
        all_pairs_so_far = list(zip(per_sample, pairs[: len(per_sample)]))
        for s, pair in all_pairs_so_far:
            cat = s["category"]
            if cat not in seen_cats or cat == "unknown":
                if cat not in seen_cats:
                    seen_cats.add(cat)
                    export_pairs.append(pair)
            if len(export_pairs) >= args.export_top_k_visuals:
                break
        if len(export_pairs) < args.export_top_k_visuals:
            for _, pair in all_pairs_so_far:
                if pair not in export_pairs:
                    export_pairs.append(pair)
                if len(export_pairs) >= args.export_top_k_visuals:
                    break

        for mid, ppt, gltf in export_pairs:
            try:
                pcd = load_points(ppt)
                if pcd.shape[0] > args.n_points:
                    idx = torch.randperm(pcd.shape[0])[: args.n_points]
                    pcd = pcd[idx]
                pcd_n = normalize(pcd).numpy()
                mesh = load_mesh_concat(gltf)
                mpts = sample_mesh_points(mesh, args.n_points, rng)
                mpts_n = normalize(torch.from_numpy(mpts)).numpy()

                pcd_rot = (pcd_n @ best_R.T).astype(np.float32)
                # red = rotated point cloud, blue = mesh sample
                pc_colored = np.concatenate(
                    [
                        np.concatenate([pcd_rot, np.tile([220, 20, 60], (pcd_rot.shape[0], 1))], axis=1),
                        np.concatenate([mpts_n, np.tile([20, 60, 220], (mpts_n.shape[0], 1))], axis=1),
                    ],
                    axis=0,
                )
                cloud = trimesh.points.PointCloud(
                    vertices=pc_colored[:, :3].astype(np.float32),
                    colors=pc_colored[:, 3:].astype(np.uint8),
                )
                cloud.export(vis_dir / f"{mid}_aligned.ply")
            except Exception as exc:
                print(f"[warn][vis] {mid}: {exc}")

    print("\n[done]")


if __name__ == "__main__":
    main()
