#!/usr/bin/env python3
"""List all object directories under 3DCoMPaT200 eval layout.

Scans {data_root}/{category}/{coarse_b'uid'|fine_b'uid'}/ and writes one path
per line. Used as input for GLB prep and parts_partsam conversion.

Usage:
    python pipes_tools/list_d3compat200_objects.py \
        --data_root /path/to/3dcompat200 \
        --out_txt /path/to/test_obj_list.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_txt", type=Path, required=True)
    ap.add_argument("--require_points", action="store_true",
                    help="skip dirs without points.pt")
    args = ap.parse_args()

    paths: list[str] = []
    for cat_dir in sorted(args.data_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for obj_dir in sorted(cat_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            if args.require_points and not (obj_dir / "points.pt").exists():
                continue
            paths.append(str(obj_dir))

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text("\n".join(paths) + ("\n" if paths else ""))
    print(f"Wrote {len(paths)} paths -> {args.out_txt}")


if __name__ == "__main__":
    main()
