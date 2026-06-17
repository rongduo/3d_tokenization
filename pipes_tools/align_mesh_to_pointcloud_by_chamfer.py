#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    raise ImportError("Please install trimesh first: pip install trimesh") from exc


DEFAULT_POINT_PATH = (
    "/x2robot_v2/lanejin/new_data/cosmo3d/dataset/datasets/datasets/test/"
    "3dcompat200/airplane/coarse_b'00_004'/points.pt"
)
DEFAULT_MESH_PATH = (
    "/x2robot_v2/lanejin/new_data/cosmo3d/dataset/datasets/datasets/train/3dcompat/"
    "datasets--CoMPaT--3DCoMPaT200/snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/"
    "models/00_004.gltf"
)
DEFAULT_OUT_DIR = "/x2robot_v2/lanejin/new_data/cosmo3d/results/aligntest"


def load_points_tensor(points_path: Path) -> torch.Tensor:
    data = torch.load(points_path, map_location="cpu")
    pts = None

    if isinstance(data, torch.Tensor):
        pts = data
    elif isinstance(data, np.ndarray):
        pts = torch.from_numpy(data)
    elif isinstance(data, dict):
        for key in ["points", "xyz", "pointcloud", "pcd", "verts", "vertices"]:
            if key in data:
                value = data[key]
                if isinstance(value, torch.Tensor):
                    pts = value
                elif isinstance(value, np.ndarray):
                    pts = torch.from_numpy(value)
                break
        if pts is None:
            for value in data.values():
                if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == 3:
                    pts = value
                    break
    elif isinstance(data, (list, tuple)):
        arr = np.asarray(data)
        if arr.ndim >= 2 and arr.shape[-1] == 3:
            pts = torch.from_numpy(arr)

    if pts is None:
        raise ValueError(f"Unsupported points.pt format: {type(data)}")

    pts = pts.float()
    if pts.ndim == 3:
        pts = pts[0]
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"Point cloud shape should be [N,3], got {tuple(pts.shape)}")
    return pts.contiguous()


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(mesh_path, force="scene")
    if isinstance(mesh_or_scene, trimesh.Scene):
        if not mesh_or_scene.geometry:
            raise ValueError(f"No geometry in mesh scene: {mesh_path}")
        # Keep per-node transforms from GLTF scene graph before merging.
        merged = mesh_or_scene.to_geometry()
        if isinstance(merged, trimesh.Trimesh):
            return merged
        if isinstance(merged, trimesh.Scene):
            if not merged.geometry:
                raise ValueError(f"Empty geometry after scene conversion: {mesh_path}")
            return trimesh.util.concatenate(tuple(merged.geometry.values()))
        raise ValueError(f"Unexpected scene geometry type: {type(merged)}")
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    raise ValueError(f"Unsupported mesh type: {type(mesh_or_scene)}")


def sample_mesh_points(mesh: trimesh.Trimesh, n_points: int, seed: int) -> torch.Tensor:
    np.random.seed(seed)
    sampled, _ = trimesh.sample.sample_surface(mesh, count=n_points)
    return torch.from_numpy(sampled.astype(np.float32))


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    # Symmetric Chamfer Distance using squared L2
    dmat = torch.cdist(a, b, p=2)
    cd = (dmat.min(dim=1).values.pow(2).mean() + dmat.min(dim=0).values.pow(2).mean()).item()
    return float(cd)


def all_24_axis_rotations() -> List[Tuple[str, np.ndarray]]:
    rotations: List[Tuple[str, np.ndarray]] = []
    axes = np.eye(3, dtype=np.float32)
    for perm in itertools.permutations([0, 1, 2]):
        base = axes[list(perm)]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            m = base.copy()
            for i in range(3):
                m[i] *= signs[i]
            # Use row vectors: p_rot = p @ R.T, so det(R)=1 for proper rotation.
            if np.linalg.det(m) > 0.99:
                name = (
                    f"perm{perm}_signs({int(signs[0])},{int(signs[1])},{int(signs[2])})"
                )
                rotations.append((name, m))
    return rotations


def export_xyz(path: Path, points: np.ndarray) -> None:
    np.savetxt(path, points, fmt="%.6f")


def export_ply(path: Path, points: np.ndarray) -> None:
    point_cloud = trimesh.points.PointCloud(vertices=points.astype(np.float32))
    point_cloud.export(path)


def rotate_points(points: torch.Tensor, r: np.ndarray) -> torch.Tensor:
    r_t = torch.from_numpy(r).float()
    return points @ r_t.T


def export_mesh_as_obj_no_texture(mesh: trimesh.Trimesh, path: Path) -> None:
    mesh_to_export = mesh.copy()
    mesh_to_export.visual = trimesh.visual.ColorVisuals(mesh_to_export)
    obj_text = trimesh.exchange.obj.export_obj(mesh_to_export, include_texture=False)
    path.write_text(obj_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find mesh rotation that best aligns with point cloud by Chamfer Distance."
    )
    parser.add_argument("--points", type=str, default=DEFAULT_POINT_PATH, help="Path to points.pt")
    parser.add_argument("--mesh", type=str, default=DEFAULT_MESH_PATH, help="Path to mesh (.gltf/.glb)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--sample_points",
        type=int,
        default=4096,
        help="Number of points sampled from mesh for chamfer",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    points_path = Path(args.points)
    mesh_path = Path(args.mesh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pc = load_points_tensor(points_path)
    if pc.shape[0] > args.sample_points:
        idx = torch.randperm(pc.shape[0])[: args.sample_points]
        pc = pc[idx]
    pc = pc.float().contiguous()

    mesh = load_mesh(mesh_path)
    mesh_pts = sample_mesh_points(mesh, n_points=pc.shape[0], seed=args.seed)

    before_cd = chamfer_distance(mesh_pts, pc)
    candidates = all_24_axis_rotations()

    best: Dict[str, object] = {
        "name": "identity",
        "rotation_matrix": np.eye(3, dtype=np.float32),
        "pointcloud_rotation_matrix": np.eye(3, dtype=np.float32),
        "translation": np.zeros(3, dtype=np.float32),
        "chamfer": before_cd,
    }
    all_scores = []

    for name, r in candidates:
        # pcd' = pcd @ R.T, mesh fixed, strictly no translation.
        rotated_pc = rotate_points(pc, r)
        cd = chamfer_distance(mesh_pts, rotated_pc)
        all_scores.append(
            {
                "name": name,
                "chamfer": cd,
                "rotation_matrix": r.tolist(),
                "translation": [0.0, 0.0, 0.0],
            }
        )
        if cd < float(best["chamfer"]):
            best = {
                "name": name,
                "rotation_matrix": r,
                "pointcloud_rotation_matrix": r,
                "translation": np.zeros(3, dtype=np.float32),
                "chamfer": cd,
            }

    best_r = np.asarray(best["rotation_matrix"], dtype=np.float32)
    best_t = np.asarray(best["translation"], dtype=np.float32)
    pc_before_aligned_np = pc.numpy().astype(np.float32)
    pc_aligned = rotate_points(pc, best_r).numpy().astype(np.float32)

    export_xyz(out_dir / "mesh_points.xyz", mesh_pts.numpy())
    export_xyz(out_dir / "pointcloud_points_before.xyz", pc_before_aligned_np)
    export_xyz(out_dir / "pointcloud_points_aligned.xyz", pc_aligned)
    export_ply(out_dir / "pointcloud_points_before.ply", pc_before_aligned_np)
    export_ply(out_dir / "pointcloud_points_aligned.ply", pc_aligned)

    export_mesh_as_obj_no_texture(mesh, out_dir / "mesh_fixed.obj")

    result = {
        "points_path": str(points_path),
        "mesh_path": str(mesh_path),
        "sampled_points": int(pc.shape[0]),
        "chamfer_before": before_cd,
        "best_rotation_name": str(best["name"]),
        "best_rotation_matrix": best_r.tolist(),
        "best_pointcloud_rotation_matrix": best_r.tolist(),
        "best_translation": best_t.tolist(),
        "best_chamfer": float(best["chamfer"]),
        "improvement": float(before_cd - float(best["chamfer"])),
        "all_scores_sorted": sorted(all_scores, key=lambda x: x["chamfer"]),
    }

    with open(out_dir / "alignment_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== Alignment Finished ===")
    print(f"Points: {points_path}")
    print(f"Mesh: {mesh_path}")
    print(f"Output dir: {out_dir}")
    print(f"Chamfer before: {before_cd:.6f}")
    print(f"Best rotation: {best['name']}")
    print("Best rotation matrix:")
    print(best_r)
    print(f"Best translation: {best_t.tolist()}")
    print(f"Best chamfer: {float(best['chamfer']):.6f}")
    print(f"Improvement: {before_cd - float(best['chamfer']):.6f}")


if __name__ == "__main__":
    main()
