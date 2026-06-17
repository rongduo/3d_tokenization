
"""
Quick check:
1) load one 3dcompat training object
2) extract Utonia features
3) compare point counts and coordinate consistency

Run:
python -m pipes.ours_token3d._test
"""

from pathlib import Path
import sys
import importlib.util

import numpy as np
import torch


def _load_pt(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    data_dir = Path(
        "/x2robot_v2/lanejin/new_data/cosmo3d/dataset/datasets/datasets/train/3dcompat/forfind3dtrain/fine_b'0d_018'"
    )

    # Add Utonia repo to PYTHONPATH for local import.
    utonia_root = Path("/x2robot_v2/lanejin/new_data/Utonia")
    if str(utonia_root) not in sys.path:
        sys.path.insert(0, str(utonia_root))

    import utonia  # type: ignore
    # Avoid "pipes" namespace collision with current repo by loading from absolute file path.
    copule_path = utonia_root / "pipes" / "exfeats" / "feature_pca_copule.py"
    spec = importlib.util.spec_from_file_location("utonia_feature_pca_copule", str(copule_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {copule_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    init_utonia_model = mod.init_utonia_model

    pts_xyz = _load_pt(data_dir / "points.pt").float().cpu()  # [N, 3]
    pts_rgb = _load_pt(data_dir / "rgb.pt").float().cpu()      # [N, 3], usually 0~1
    normal = _load_pt(data_dir / "normals.pt").float().cpu()   # [N, 3]

    # Utonia expects numpy dict: coord/color/normal.
    sample = {
        "coord": pts_xyz.numpy().astype(np.float32),
        "color": pts_rgb.numpy().astype(np.float32),
        "normal": normal.numpy().astype(np.float32),
    }

    model, device = init_utonia_model()
    transform = utonia.transform.default()
    point = transform(sample)
    sampled_coord = point["coord"].detach().cpu().float().clone()

    with torch.inference_mode():
        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor) and device == "cuda":
                point[k] = point[k].cuda(non_blocking=True)
        out = model(point)
        # Match feature_pca_copule.featurize_pca_to_ply:
        # propagate features back through pooling hierarchy.
        for _ in range(2):
            assert "pooling_parent" in out.keys()
            assert "pooling_inverse" in out.keys()
            parent = out.pop("pooling_parent")
            inverse = out.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
            out = parent
        while "pooling_parent" in out.keys():
            assert "pooling_inverse" in out.keys()
            parent = out.pop("pooling_parent")
            inverse = out.pop("pooling_inverse")
            parent.feat = out.feat[inverse]
            out = parent

        utonia_coord = out.coord.detach().cpu().float()
        utonia_feat = out.feat.detach().cpu().float()

    n_raw = int(pts_xyz.shape[0])
    n_sampled = int(sampled_coord.shape[0])
    n_utonia = int(utonia_coord.shape[0])
    same_count = n_raw == n_utonia
    sampled_same_count = n_sampled == n_utonia

    print(f"raw points count      : {n_raw}")
    print(f"sampled points count  : {n_sampled}")
    print(f"utonia coord count    : {n_utonia}")
    print(f"count exactly equal   : {same_count}")
    print(f"sampled==output count : {sampled_same_count}")
    print(f"utonia feat shape     : {tuple(utonia_feat.shape)}")
    print(f"utonia coord shape    : {tuple(utonia_coord.shape)}")

    if sampled_same_count:
        sampled_allclose = torch.allclose(utonia_coord, sampled_coord, atol=1e-6, rtol=1e-5)
        sampled_max_abs_diff = (utonia_coord - sampled_coord).abs().max().item()
        print(f"coord allclose(sample): {sampled_allclose}")
        print(f"coord max diff(sample): {sampled_max_abs_diff:.6e}")
    else:
        print("coord allclose(sample): skipped (count mismatch)")

    if same_count:
        # Check whether coordinate values are also identical (or only count matches).
        is_allclose = torch.allclose(utonia_coord, pts_xyz, atol=1e-6, rtol=1e-5)
        max_abs_diff = (utonia_coord - pts_xyz).abs().max().item()
        print(f"coord allclose(raw)   : {is_allclose}")
        print(f"coord max abs diff    : {max_abs_diff:.6e}")
    else:
        print("coord allclose(raw)   : skipped (count mismatch)")


if __name__ == "__main__":
    main()



