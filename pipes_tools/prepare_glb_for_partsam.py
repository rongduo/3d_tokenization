#!/usr/bin/env python3
"""Batch convert 3DCoMPaT GLTF meshes to GLB for PartSAM inference.

Reads train.txt, extracts UIDs from directory names, converts corresponding
GLTF meshes to single GLB files (concatenating scene nodes).

Usage:
    python pipes_tools/prepare_glb_for_partsam.py \
        --train_txt /path/to/d3compat/train.txt \
        --mesh_root /data3/jl/dataset/.../Compat200/models \
        --out_dir /data5/jl/project/PartSAM/compat_eval/glb_train \
        --limit 100
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import trimesh


def concat_scene(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(path), process=False, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        return scene
    parts: List[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        tf, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue
        m = geom.copy()
        m.apply_transform(tf)
        parts.append(m)
    if not parts:
        raise RuntimeError(f"no Trimesh parts in {path}")
    return trimesh.util.concatenate(parts)


def parse_uid_from_dirname(name: str) -> str | None:
    m = re.search(r"'([^']+)'", name)
    if m:
        return m.group(1)
    if name.startswith("coarse_b"):
        return name.replace("coarse_b", "").strip("_")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", type=Path, required=True,
                    help="path to train.txt with object directory paths")
    ap.add_argument("--mesh_root", type=Path, required=True,
                    help="directory with {uid}.gltf files")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=-1,
                    help="process first N objects, -1 for all")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    obj_paths = args.train_txt.read_text().splitlines()
    if args.limit > 0:
        obj_paths = obj_paths[:args.limit]

    # Extract unique UIDs (multiple coarse/fine variants can share same mesh)
    seen_uids = set()
    uids = []
    for obj_path in obj_paths:
        uid = parse_uid_from_dirname(Path(obj_path).name)
        if uid and uid not in seen_uids:
            seen_uids.add(uid)
            uids.append(uid)

    print(f"Objects: {len(obj_paths)}, unique UIDs: {len(uids)}")

    ok, skip, fail = 0, 0, 0
    for i, uid in enumerate(uids):
        src = args.mesh_root / f"{uid}.gltf"
        out = args.out_dir / f"{uid}.glb"
        if out.exists():
            skip += 1
            if i < 10:
                print(f"[{i+1}/{len(uids)}] {uid}: skip (exists)")
            continue
        if not src.exists():
            fail += 1
            print(f"[{i+1}/{len(uids)}] {uid}: GLTF not found")
            continue
        try:
            mesh = concat_scene(src)
            mesh.export(out)
            ok += 1
            if i < 10 or i % 500 == 0:
                print(f"[{i+1}/{len(uids)}] {uid}: {len(mesh.faces)} faces -> {out}")
        except Exception as exc:
            fail += 1
            print(f"[{i+1}/{len(uids)}] {uid}: FAIL ({exc})")

    print(f"\nDone. {ok} converted, {skip} skipped, {fail} failed. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
