#!/usr/bin/env python3
"""Split GLTF symlinks into N shards for multi-GPU PartSAM inference.

Training used compat_eval/gltf_gpu{0..6}/ with ~2292 disjoint .gltf each.
This script partitions gltf_test_all into gltf_test_gpu{i}/ shards.

Usage:
    python pipes_tools/split_gltf_for_multigpu.py \
        --src_dir /path/to/gltf_test_all \
        --out_root /path/to/PartSAM/compat_eval \
        --num_shards 7 \
        --prefix gltf_test_gpu
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=Path, required=True,
                    help="directory with all {uid}.gltf symlinks")
    ap.add_argument("--out_root", type=Path, required=True,
                    help="parent dir for gltf_test_gpu0, gltf_test_gpu1, ...")
    ap.add_argument("--num_shards", type=int, default=7)
    ap.add_argument("--prefix", type=str, default="gltf_test_gpu")
    ap.add_argument("--clean", action="store_true",
                    help="remove existing shard dirs before splitting")
    args = ap.parse_args()

    gltfs = sorted(args.src_dir.glob("*.gltf"))
    if not gltfs:
        raise SystemExit(f"No .gltf in {args.src_dir}")

    n = len(gltfs)
    k = args.num_shards
    chunk = math.ceil(n / k)

    if args.clean:
        for i in range(k):
            shard = args.out_root / f"{args.prefix}{i}"
            if shard.exists():
                for p in shard.glob("*.gltf"):
                    p.unlink()
                shard.rmdir()

    total_linked = 0
    for i in range(k):
        shard_dir = args.out_root / f"{args.prefix}{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        subset = gltfs[i * chunk : (i + 1) * chunk]
        linked = 0
        for src in subset:
            dst = shard_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            linked += 1
        total_linked += linked
        print(f"shard {i}: {linked} gltfs -> {shard_dir}")

    print(f"\nSplit {n} gltfs into {k} shards ({total_linked} links)")


if __name__ == "__main__":
    main()
