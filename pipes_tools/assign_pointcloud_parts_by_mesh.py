#!/usr/bin/env python3
import argparse
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
DEFAULT_OUT_DIR = "/x2robot_v2/lanejin/new_data/cosmo3d/results/aligntest/part_assign"

# Confirmed rotation from your alignment check:
# [[ 1, 0, 0],
#  [ 0, 0, 1],
#  [ 0,-1, 0]]
DEFAULT_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


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


def rotate_points(points: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (points @ r.T).astype(np.float32)


def fine_part_type(node_name: str) -> str:
    """Strip glTF instance suffix: airplane_door_01 -> airplane_door."""
    import re

    name = str(node_name)
    m = re.match(r"(.+?)_\d+$", name)
    return m.group(1) if m else name


def _as_trimesh_list_with_part_name(loaded) -> List[Tuple[str, trimesh.Trimesh]]:
    # Reference idea from PartSAM/pipes/seglab_compat/s3_maskparts.py:
    # use scene graph node names as part units (not geometry names like Mesh.001).
    out: List[Tuple[str, trimesh.Trimesh]] = []

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return out
        # Iterate scene graph nodes so transforms are preserved.
        for node_name in loaded.graph.nodes_geometry:
            tf, geom_name = loaded.graph[node_name]
            if geom_name not in loaded.geometry:
                continue
            geom = loaded.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh):
                continue
            mesh_copy = geom.copy()
            mesh_copy.apply_transform(tf)
            part_name = str(node_name)
            out.append((part_name, mesh_copy))
        # Fallback when nodes_geometry is empty
        if not out:
            for geom_name, geom in loaded.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    out.append((str(geom_name), geom.copy()))
    elif isinstance(loaded, trimesh.Trimesh):
        out.append(("mesh", loaded.copy()))
    else:
        raise RuntimeError(f"Unsupported mesh type: {type(loaded)}")
    return out


def load_mesh_fine_part_gt(
    mesh_path: Path,
    *,
    merge_instances: bool = True,
) -> Tuple[trimesh.Trimesh, np.ndarray, List[str]]:
    """Load glTF mesh with per-face fine-part GT from scene node names.

    Args:
        merge_instances: If True, merge airplane_door_01/02 into one fine part.
    """
    loaded = trimesh.load(str(mesh_path), process=False, force="scene")
    part_meshes = _as_trimesh_list_with_part_name(loaded)
    if not part_meshes:
        raise RuntimeError(f"No valid mesh parts found in: {mesh_path}")

    merged_mesh_list: List[trimesh.Trimesh] = []
    face_to_part_chunks: List[np.ndarray] = []
    part_names: List[str] = []
    part_name_to_idx: Dict[str, int] = {}

    for node_name, part_mesh in part_meshes:
        n_faces = int(len(part_mesh.faces))
        if n_faces <= 0:
            continue

        label_name = fine_part_type(node_name) if merge_instances else str(node_name)
        if label_name.startswith("__"):
            continue

        if label_name not in part_name_to_idx:
            part_name_to_idx[label_name] = len(part_names)
            part_names.append(label_name)
        pidx = part_name_to_idx[label_name]

        merged_mesh_list.append(part_mesh)
        face_to_part_chunks.append(np.full((n_faces,), pidx, dtype=np.int32))

    if not merged_mesh_list:
        raise RuntimeError(f"All parts have zero faces: {mesh_path}")

    merged_mesh = trimesh.util.concatenate(tuple(merged_mesh_list))
    face_to_part = np.concatenate(face_to_part_chunks, axis=0)
    return merged_mesh, face_to_part, part_names


def load_mesh_with_face_part_map(mesh_path: Path) -> Tuple[trimesh.Trimesh, np.ndarray, List[str]]:
    loaded = trimesh.load(str(mesh_path), process=False, force="scene")
    part_meshes = _as_trimesh_list_with_part_name(loaded)
    if not part_meshes:
        raise RuntimeError(f"No valid mesh parts found in: {mesh_path}")

    merged_mesh_list: List[trimesh.Trimesh] = []
    face_to_part_chunks: List[np.ndarray] = []
    part_names: List[str] = []
    part_name_to_idx: Dict[str, int] = {}

    for part_name, part_mesh in part_meshes:
        n_faces = int(len(part_mesh.faces))
        if n_faces <= 0:
            continue

        if part_name not in part_name_to_idx:
            part_name_to_idx[part_name] = len(part_names)
            part_names.append(part_name)
        pidx = part_name_to_idx[part_name]

        merged_mesh_list.append(part_mesh)
        face_to_part_chunks.append(np.full((n_faces,), pidx, dtype=np.int32))

    if not merged_mesh_list:
        raise RuntimeError(f"All parts have zero faces: {mesh_path}")

    merged_mesh = trimesh.util.concatenate(tuple(merged_mesh_list))
    face_to_part = np.concatenate(face_to_part_chunks, axis=0)
    return merged_mesh, face_to_part, part_names


def build_palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.integers(30, 256, size=(max(n, 1), 3), dtype=np.uint8)
    colors[0] = np.array([220, 20, 60], dtype=np.uint8)
    return colors


def save_part_labels_txt(path: Path, labels: np.ndarray, part_names: List[str]) -> None:
    lines = ["point_index\tpart_index\tpart_name"]
    for i, p in enumerate(labels.tolist()):
        pname = part_names[p] if 0 <= p < len(part_names) else "unknown"
        lines.append(f"{i}\t{p}\t{pname}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_part_count_summary(path: Path, labels: np.ndarray, part_names: List[str]) -> None:
    uniq, cnt = np.unique(labels, return_counts=True)
    lines = ["part_index\tpart_name\tpoint_count"]
    for p, c in zip(uniq.tolist(), cnt.tolist()):
        pname = part_names[p] if 0 <= p < len(part_names) else "unknown"
        lines.append(f"{p}\t{pname}\t{c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_mesh_as_obj_no_texture(mesh: trimesh.Trimesh, path: Path) -> None:
    mesh_to_export = mesh.copy()
    mesh_to_export.visual = trimesh.visual.ColorVisuals(mesh_to_export)
    obj_text = trimesh.exchange.obj.export_obj(mesh_to_export, include_texture=False)
    path.write_text(obj_text, encoding="utf-8")


def assign_parts_by_nearest_face(mesh: trimesh.Trimesh, face_to_part: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Prefer exact closest-point query; fallback to nearest face-centroid if rtree is unavailable.
    try:
        _, _, face_idx = trimesh.proximity.closest_point(mesh, points)
        face_idx = face_idx.astype(np.int64)
    except Exception:
        tri_centers = np.asarray(mesh.triangles_center, dtype=np.float32)
        if tri_centers.shape[0] != face_to_part.shape[0]:
            raise RuntimeError(
                f"face count mismatch: centers={tri_centers.shape[0]} vs map={face_to_part.shape[0]}"
            )
        n_points = points.shape[0]
        face_idx = np.empty((n_points,), dtype=np.int64)
        chunk = 2048
        for s in range(0, n_points, chunk):
            e = min(s + chunk, n_points)
            # squared distance, shape [chunk, num_faces]
            diff = points[s:e, None, :] - tri_centers[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            face_idx[s:e] = np.argmin(d2, axis=1)

    valid = (face_idx >= 0) & (face_idx < len(face_to_part))
    labels = np.full((points.shape[0],), -1, dtype=np.int32)
    labels[valid] = face_to_part[face_idx[valid]]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align point cloud to mesh by fixed rotation and assign part labels."
    )
    parser.add_argument("--points", type=str, default=DEFAULT_POINT_PATH, help="Path to points.pt")
    parser.add_argument("--mesh", type=str, default=DEFAULT_MESH_PATH, help="Path to mesh (.gltf/.glb)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--rotation",
        type=float,
        nargs=9,
        default=DEFAULT_ROTATION.reshape(-1).tolist(),
        help="Row-major 3x3 rotation matrix for point cloud alignment.",
    )
    args = parser.parse_args()

    points_path = Path(args.points)
    mesh_path = Path(args.mesh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r = np.asarray(args.rotation, dtype=np.float32).reshape(3, 3)
    pc = load_points_tensor(points_path).numpy().astype(np.float32)
    pc_aligned = rotate_points(pc, r)

    mesh, face_to_part, part_names = load_mesh_with_face_part_map(mesh_path)

    # Assign each point to mesh part.
    part_labels = assign_parts_by_nearest_face(mesh, face_to_part, pc_aligned)

    colors = build_palette(len(part_names), seed=42)
    point_colors = np.zeros((pc_aligned.shape[0], 3), dtype=np.uint8)
    unknown_color = np.array([120, 120, 120], dtype=np.uint8)
    point_colors[:] = unknown_color
    known = part_labels >= 0
    point_colors[known] = colors[part_labels[known]]

    colored_pc = trimesh.points.PointCloud(vertices=pc_aligned, colors=point_colors)
    colored_pc.export(out_dir / "pointcloud_part_colored.ply")
    trimesh.points.PointCloud(vertices=pc_aligned).export(out_dir / "pointcloud_aligned.ply")
    export_mesh_as_obj_no_texture(mesh, out_dir / "mesh_fixed.obj")

    np.save(out_dir / "pointcloud_part_labels.npy", part_labels)
    save_part_labels_txt(out_dir / "pointcloud_part_labels.txt", part_labels, part_names)
    save_part_count_summary(out_dir / "part_point_count_summary.txt", part_labels, part_names)

    part_map_lines = ["part_index\tpart_name\trgb"]
    for i, name in enumerate(part_names):
        c = colors[i]
        part_map_lines.append(f"{i}\t{name}\t({int(c[0])},{int(c[1])},{int(c[2])})")
    (out_dir / "part_color_mapping.txt").write_text("\n".join(part_map_lines) + "\n", encoding="utf-8")

    print("=== Part Assignment Finished ===")
    print(f"Points: {points_path}")
    print(f"Mesh: {mesh_path}")
    print(f"Output dir: {out_dir}")
    print("Point-cloud rotation matrix:")
    print(r)
    print(f"Num points: {pc_aligned.shape[0]}")
    print(f"Num parts: {len(part_names)}")
    print("Saved: pointcloud_part_labels.npy/txt, pointcloud_part_colored.ply")


if __name__ == "__main__":
    main()
