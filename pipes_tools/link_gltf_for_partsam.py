#!/usr/bin/env python3
"""Symlink test/train GLTF meshes for PartSAM inference (no GLB conversion).

PartSAM ValDataset accepts .gltf directly. Training inference used
compat_eval/gltf_gpu{0..6}/ with symlinks to source {uid}.gltf files.

Usage:
    python pipes_tools/link_gltf_for_partsam.py \
        --obj_list /path/to/test_obj_list.txt \
        --mesh_root /path/to/Compat200/models \
        --out_dir /path/to/gltf_test_all
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_uid(name: str) -> str | None:
    m = re.search(r"'([^']+)'", name)
    if m:
        return m.group(1)
    if name.startswith("coarse_b") or name.startswith("fine_b"):
        return name.split("b", 1)[-1].strip("_")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_list", type=Path, required=True)
    ap.add_argument("--mesh_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    uids: list[str] = []
    for line in args.obj_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        uid = parse_uid(Path(line).name)
        if uid and uid not in seen:
            seen.add(uid)
            uids.append(uid)
    if args.limit > 0:
        uids = uids[: args.limit]

    ok = skip = fail = 0
    for i, uid in enumerate(uids):
        src = args.mesh_root / f"{uid}.gltf"
        dst = args.out_dir / f"{uid}.gltf"
        if dst.exists() or dst.is_symlink():
            skip += 1
            continue
        if not src.exists():
            fail += 1
            if fail <= 10:
                print(f"[missing] {uid}: {src}")
            continue
        dst.symlink_to(src.resolve())
        ok += 1
        if ok <= 5 or ok % 500 == 0:
            print(f"[{i+1}/{len(uids)}] linked {uid}.gltf")

    print(f"\nDone. linked={ok} skip={skip} missing={fail} unique_uids={len(uids)}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
