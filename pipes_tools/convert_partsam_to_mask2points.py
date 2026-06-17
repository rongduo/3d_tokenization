#!/usr/bin/env python3
"""Convert PartSAM PLY output to Token3D training data format.

For each object:
  1. Look up object category -> category-specific rotation matrix
  2. Load PartSAM's colored PLY -> face-level part labels (via face colors)
  3. Load the original training point cloud (points.pt)
  4. Rotate points with category-specific rotation to align with mesh frame
  5. Map each point to nearest mesh face -> assign PartSAM part label
  6. Generate mask2points.pt (binary mask matrix) and mask_labels.txt

Usage:
    python pipes_tools/convert_partsam_to_mask2points.py \
        --train_txt /path/to/d3compat/train.txt \
        --results_dir /path/to/PartSAM/results \
        --category_mapping /path/to/train_object_category_mapping.json \
        --rotation_dict /path/to/category_canonical_to_glb_rot_dict.json \
        --limit 10
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import trimesh

# Fallback rotation for categories not found in rotation_dict.
DEFAULT_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def parse_uid_from_dirname(name: str) -> str | None:
    m = re.search(r"'([^']+)'", name)
    if m:
        return m.group(1)
    if name.startswith("coarse_b"):
        return name.replace("coarse_b", "").strip("_")
    return None


def load_category_mapping(path: Path) -> Dict[str, str]:
    """Load {uid: category_name} mapping."""
    with open(path, "r") as f:
        return json.load(f)


def load_rotation_dict(path: Path) -> Dict[str, np.ndarray]:
    """Load {category_name: 3x3_rotation_matrix} mapping."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {cat: np.array(mat, dtype=np.float32) for cat, mat in raw.items()}


def load_partsam_face_labels(ply_path: Path) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Load PartSAM PLY and extract integer face labels from face colors."""
    mesh = trimesh.load(str(ply_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    fc = np.asarray(mesh.visual.face_colors)[:, :3]
    # Map each unique RGB to an integer label; black (0,0,0) = background (-1)
    uniq, inv = np.unique(fc, axis=0, return_inverse=True)
    labels = inv.astype(np.int64)
    black = np.where(np.all(uniq == 0, axis=1))[0]
    if len(black):
        labels[labels == black[0]] = -1
    return mesh, labels


def rotate_points(points: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (points @ r.T).astype(np.float32)


def assign_points_to_faces(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Assign each point to the nearest mesh face, return per-point part labels."""
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

    valid = (face_idx >= 0) & (face_idx < len(face_labels))
    pt_labels = np.full((points.shape[0],), -1, dtype=np.int64)
    pt_labels[valid] = face_labels[face_idx[valid]]
    return pt_labels


def build_mask2points(pt_labels: np.ndarray) -> Tuple[torch.Tensor, List[str]]:
    """Convert per-point part labels to mask2points binary matrix and part names."""
    unique_parts = sorted(np.unique(pt_labels[pt_labels >= 0]))
    if len(unique_parts) == 0:
        raise RuntimeError("No valid part labels found")

    n_points = len(pt_labels)
    n_parts = len(unique_parts)
    mask = torch.zeros((n_parts, n_points), dtype=torch.float32)
    part_names = []
    for i, p in enumerate(unique_parts):
        mask[i, pt_labels == p] = 1.0
        part_names.append(f"part_{i}")

    return mask, part_names


def process_one_object(
    obj_dir: Path,
    ply_path: Path,
    rotation: np.ndarray,
    suffix: str = "_partsam",
) -> Tuple[bool, str]:
    """Convert one object: PartSAM PLY -> mask2points.pt + mask_labels.txt + parts.pt."""
    points_path = obj_dir / "points.pt"
    if not points_path.exists():
        return False, "missing points.pt"

    if not ply_path.exists():
        return False, f"missing PartSAM PLY: {ply_path}"

    # Load and rotate point cloud
    pts_tensor = torch.load(points_path, map_location="cpu")
    if isinstance(pts_tensor, torch.Tensor):
        pts = pts_tensor.float().cpu().numpy()
    else:
        pts = np.asarray(pts_tensor, dtype=np.float32)
    if pts.ndim == 3:
        pts = pts[0]
    pts_rot = rotate_points(pts, rotation)

    # Load PartSAM face labels
    mesh, face_labels = load_partsam_face_labels(ply_path)

    # Assign each point to nearest face
    pt_labels = assign_points_to_faces(mesh, face_labels, pts_rot)

    # Build mask2points
    mask, part_names = build_mask2points(pt_labels)

    # Save with suffix to avoid overwriting original files
    torch.save(mask, obj_dir / f"mask2points{suffix}.pt")
    torch.save(torch.from_numpy(pt_labels.astype(np.int32)), obj_dir / f"parts{suffix}.pt")
    (obj_dir / f"mask_labels{suffix}.txt").write_text("\n".join(part_names) + "\n")

    n_pts_assigned = int((pt_labels >= 0).sum())
    return True, f"parts={len(part_names)} assigned={n_pts_assigned}/{len(pt_labels)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", type=Path, required=True,
                    help="path to train.txt listing object directories")
    ap.add_argument("--results_dir", type=Path, required=True,
                    help="directory with PartSAM {uid}.ply outputs")
    ap.add_argument("--category_mapping", type=Path, required=True,
                    help="train_object_category_mapping.json: uid -> category name")
    ap.add_argument("--rotation_dict", type=Path, required=True,
                    help="category_canonical_to_glb_rot_dict.json: category -> 3x3 rotation")
    ap.add_argument("--limit", type=int, default=-1,
                    help="process first N objects, -1 for all")
    ap.add_argument("--suffix", type=str, default="_partsam",
                    help="suffix for output files (mask2points{suffix}.pt, etc.)")
    args = ap.parse_args()

    cat_map = load_category_mapping(args.category_mapping)
    rot_dict = load_rotation_dict(args.rotation_dict)

    obj_paths = [Path(p) for p in args.train_txt.read_text().splitlines()]
    if args.limit > 0:
        obj_paths = obj_paths[:args.limit]

    ok, fail, no_rot = 0, 0, 0
    for i, obj_dir in enumerate(obj_paths):
        uid = parse_uid_from_dirname(obj_dir.name)
        if uid is None:
            fail += 1
            print(f"[{i+1}/{len(obj_paths)}] {obj_dir.name}: cannot parse UID")
            continue

        # Look up category-specific rotation
        # cat_map uses underscores (e.g. "bbq_grill"), rot_dict uses spaces ("bbq grill")
        category = cat_map.get(uid)
        if category is None:
            rotation = DEFAULT_ROTATION
            no_rot += 1
            rot_note = "no_category"
        elif category in rot_dict:
            rotation = rot_dict[category]
            rot_note = category
        elif category.replace("_", " ") in rot_dict:
            rotation = rot_dict[category.replace("_", " ")]
            rot_note = category
        else:
            rotation = DEFAULT_ROTATION
            no_rot += 1
            rot_note = f"no_rot_for_{category}"

        ply_path = args.results_dir / f"{uid}.ply"
        try:
            success, msg = process_one_object(obj_dir, ply_path, rotation, args.suffix)
            if success:
                ok += 1
                print(f"[{i+1}/{len(obj_paths)}] {obj_dir.name}: OK cat={rot_note} ({msg})")
            else:
                fail += 1
                print(f"[{i+1}/{len(obj_paths)}] {obj_dir.name}: SKIP ({msg})")
        except Exception as exc:
            fail += 1
            print(f"[{i+1}/{len(obj_paths)}] {obj_dir.name}: FAIL ({exc})")

    print(f"\nDone. {ok} processed, {fail} failed, {no_rot} used default rotation.")


if __name__ == "__main__":
    main()
