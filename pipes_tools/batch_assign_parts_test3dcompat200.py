#!/usr/bin/env python3
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    raise ImportError("Please install trimesh first: pip install trimesh") from exc


DEFAULT_OBJECTS_ROOT = Path(
    "/x2robot_v2/lanejin/new_data/cosmo3d/dataset/datasets/datasets/test/3dcompat200"
)
DEFAULT_MESH_ROOT = Path(
    "/x2robot_v2/lanejin/new_data/cosmo3d/dataset/datasets/datasets/train/3dcompat/"
    "datasets--CoMPaT--3DCoMPaT200/snapshots/a19e536383845527203a1f3bf0b509e12ed42bd7/models"
)

DEFAULT_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("batch_assign_parts_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def parse_uid_from_dirname(name: str) -> str | None:
    m = re.search(r"'([^']+)'", name)
    if m:
        return m.group(1)
    if name.startswith("coarse_b"):
        return name.replace("coarse_b", "").strip("_")
    return None


def load_points_tensor(points_path: Path) -> np.ndarray:
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
    return pts.contiguous().cpu().numpy().astype(np.float32)


def rotate_points(points: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (points @ r.T).astype(np.float32)


def _as_trimesh_list_with_part_name(loaded) -> List[Tuple[str, trimesh.Trimesh]]:
    out: List[Tuple[str, trimesh.Trimesh]] = []
    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            geom_name = loaded.graph[node_name][1]
            if geom_name not in loaded.geometry:
                continue
            geom = loaded.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh):
                continue
            tf = loaded.graph.get(node_name)[0]
            mesh_copy = geom.copy()
            mesh_copy.apply_transform(tf)
            out.append((str(geom_name), mesh_copy))
        if not out:
            for geom_name, geom in loaded.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    out.append((str(geom_name), geom.copy()))
    elif isinstance(loaded, trimesh.Trimesh):
        out.append(("mesh", loaded.copy()))
    else:
        raise RuntimeError(f"Unsupported mesh type: {type(loaded)}")
    return out


def load_mesh_with_face_part_map(mesh_path: Path) -> Tuple[trimesh.Trimesh, np.ndarray, List[str]]:
    loaded = trimesh.load(str(mesh_path), process=False, force="scene")
    part_meshes = _as_trimesh_list_with_part_name(loaded)
    if not part_meshes:
        raise RuntimeError(f"No valid mesh parts found in: {mesh_path}")

    merged_meshes: List[trimesh.Trimesh] = []
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
        merged_meshes.append(part_mesh)
        face_to_part_chunks.append(np.full((n_faces,), pidx, dtype=np.int32))

    if not merged_meshes:
        raise RuntimeError(f"All parts have zero faces: {mesh_path}")

    merged_mesh = trimesh.util.concatenate(tuple(merged_meshes))
    face_to_part = np.concatenate(face_to_part_chunks, axis=0)
    return merged_mesh, face_to_part, part_names


def assign_parts_by_nearest_face(mesh: trimesh.Trimesh, face_to_part: np.ndarray, points: np.ndarray) -> np.ndarray:
    try:
        _, _, face_idx = trimesh.proximity.closest_point(mesh, points)
        face_idx = face_idx.astype(np.int64)
    except Exception:
        tri_centers = np.asarray(mesh.triangles_center, dtype=np.float32)
        n_points = points.shape[0]
        face_idx = np.empty((n_points,), dtype=np.int64)
        chunk = 2048
        for s in range(0, n_points, chunk):
            e = min(s + chunk, n_points)
            diff = points[s:e, None, :] - tri_centers[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            face_idx[s:e] = np.argmin(d2, axis=1)

    valid = (face_idx >= 0) & (face_idx < len(face_to_part))
    labels = np.full((points.shape[0],), -1, dtype=np.int32)
    labels[valid] = face_to_part[face_idx[valid]]
    return labels


def build_palette(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.integers(30, 256, size=(max(n, 1), 3), dtype=np.uint8)
    colors[0] = np.array([220, 20, 60], dtype=np.uint8)
    return colors


def colorize_points(points: np.ndarray, labels: np.ndarray, n_parts: int) -> trimesh.points.PointCloud:
    colors = build_palette(n_parts)
    point_colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    point_colors[:] = np.array([120, 120, 120], dtype=np.uint8)
    known = labels >= 0
    point_colors[known] = colors[labels[known]]
    return trimesh.points.PointCloud(vertices=points, colors=point_colors)


def iter_object_dirs(objects_root: Path) -> List[Path]:
    # test/3dcompat200/<category>/<object_dir>
    out: List[Path] = []
    for cate_dir in sorted(objects_root.iterdir()):
        if not cate_dir.is_dir():
            continue
        for obj_dir in sorted(cate_dir.iterdir()):
            if obj_dir.is_dir():
                out.append(obj_dir)
    return out


def process_one_item(
    item_dir: Path,
    mesh_root: Path,
    rotation: np.ndarray,
    skip_existing: bool,
) -> Tuple[bool, str]:
    points_path = item_dir / "points.pt"
    if not points_path.exists():
        return False, "missing points.pt"

    uid = parse_uid_from_dirname(item_dir.name)
    if uid is None:
        return False, f"cannot parse uid from dirname: {item_dir.name}"

    mesh_path = mesh_root / f"{uid}.gltf"
    if not mesh_path.exists():
        return False, f"mesh not found: {mesh_path}"

    parts_out_path = item_dir / "parts.pt"
    ply_out_path = item_dir / "points_part_colored.ply"
    if skip_existing and parts_out_path.exists() and ply_out_path.exists():
        return True, "skip existing"

    points_raw = load_points_tensor(points_path)
    points_rot = rotate_points(points_raw, rotation)
    mesh, face_to_part, part_names = load_mesh_with_face_part_map(mesh_path)
    labels = assign_parts_by_nearest_face(mesh, face_to_part, points_rot)

    torch.save(torch.from_numpy(labels.astype(np.int32)), parts_out_path)
    colorize_points(points_raw, labels, len(part_names)).export(ply_out_path)
    return True, "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch assign part labels for test/3dcompat200 points.")
    parser.add_argument("--objects_root", type=Path, default=DEFAULT_OBJECTS_ROOT)
    parser.add_argument("--mesh_root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=-1, help="Process first N objects, -1 means all.")
    parser.add_argument(
        "--rotation",
        type=float,
        nargs=9,
        default=DEFAULT_ROTATION.reshape(-1).tolist(),
        help="Row-major 3x3 rotation matrix for point cloud alignment.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    log_path = script_dir / "batch_assign_parts_test3dcompat200.log"
    logger = setup_logger(log_path)

    logger.info("start batch processing")
    logger.info("objects_root=%s", str(args.objects_root))
    logger.info("mesh_root=%s", str(args.mesh_root))

    if not args.objects_root.exists():
        raise FileNotFoundError(f"objects_root not found: {args.objects_root}")
    if not args.mesh_root.exists():
        raise FileNotFoundError(f"mesh_root not found: {args.mesh_root}")

    rotation = np.asarray(args.rotation, dtype=np.float32).reshape(3, 3)
    logger.info("rotation=\n%s", str(rotation))

    obj_dirs = iter_object_dirs(args.objects_root)
    if args.limit > 0:
        obj_dirs = obj_dirs[: args.limit]
    logger.info("num_object_dirs=%d", len(obj_dirs))

    ok, skip, fail = 0, 0, 0
    for item_dir in obj_dirs:
        rel = item_dir.relative_to(args.objects_root)
        try:
            success, msg = process_one_item(
                item_dir=item_dir,
                mesh_root=args.mesh_root,
                rotation=rotation,
                skip_existing=args.skip_existing,
            )
            if success and msg == "processed":
                ok += 1
                logger.info("[done] %s", str(rel))
            elif success and msg == "skip existing":
                skip += 1
                logger.info("[skip] %s", str(rel))
            else:
                fail += 1
                logger.warning("[skip/fail] %s | %s", str(rel), msg)
        except Exception as exc:
            fail += 1
            logger.exception("[error] %s | %s", str(rel), exc)

    logger.info("summary: processed=%d skipped=%d failed=%d", ok, skip, fail)
    print(f"[summary] processed={ok} skipped={skip} failed={fail}")
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
