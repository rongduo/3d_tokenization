import os, sys, traceback
import torch
sys.path.insert(0, "/data5/hmf/Utonia")
sys.path.insert(0, ".")  # project root for pipes/...

import utonia

CKPT = "/data5/hmf/Utonia/utonia.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    import flash_attn  # noqa: F401
    model = utonia.load(CKPT, repo_id="Pointcept/Utonia").to(device)
except Exception:
    custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = utonia.load(CKPT, repo_id="Pointcept/Utonia", custom_config=custom_config).to(device)

model.eval()
print("[OK] model loaded on", device)

from pipes.ours_token3d_acc.data import TrainingData
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
ds = TrainingData("/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat")
item = ds[0]
for k, v in item.items():
    if hasattr(v, "shape"):
        print(" ", k, type(v).__name__, tuple(v.shape))
print("len item['coord']:", item["coord"].shape)

ut_in = {
    "coord": item["coord"].to(device),
    "grid_coord": item["grid_coord"].to(device),
    "offset": item["offset"].to(device),
    "feat": torch.cat([item["coord"], item["feat"]], dim=1).to(device),
}
print("input feat dim:", ut_in["feat"].shape)

with torch.inference_mode():
    out = model(ut_in)
    print("[OK] forward done. raw out.feat:", out.feat.shape)
    # replicate the training-time upsampling: 2 cat steps then propagate
    for i in range(2):
        assert "pooling_parent" in out.keys()
        parent = out.pop("pooling_parent")
        inverse = out.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
        out = parent
        print(f"  after cat step {i}: parent.feat={parent.feat.shape}")
    while "pooling_parent" in out.keys():
        parent = out.pop("pooling_parent")
        inverse = out.pop("pooling_inverse")
        parent.feat = out.feat[inverse]
        out = parent
        print(f"  after prop step: parent.feat={parent.feat.shape}")
print("FINAL utonia_feat:", out.feat.shape, "  (expect last dim == 1224)")
