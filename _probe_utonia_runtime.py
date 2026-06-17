"""Reproduce train_mutigpus.init_utonia_runtime() via the shim and run one
forward pass on a real TrainingData[0] sample, then call attach_utonia_feature
to confirm data['utonia_feat'] dim is 1224."""
import os
import sys
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("UTONIA_ROOT", "/data5/jl/project/tokenizer_seg/_utonia_root")
os.environ.setdefault("UTONIA_CKPT_DIR", "/data5/jl/project/tokenizer_seg/_utonia_root")
sys.path.insert(0, ".")

from pipes.ours_token3d_acc.train_mutigpus import init_utonia_runtime, attach_utonia_feature
from pipes.ours_token3d_acc.data import TrainingData

print("Loading Utonia via init_utonia_runtime(rank=0, enable_online_utonia=True) ...")
utonia_model, utonia_device = init_utonia_runtime(rank=0, enable_online_utonia=True)
print("OK utonia_model on", utonia_device, type(utonia_model).__name__)

print("Loading TrainingData[0] ...")
ds = TrainingData("/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat")
item = ds[0]
print("ds[0] coord:", item["coord"].shape, "feat:", item["feat"].shape)

# Move tensors to the same device as the rest of training would (cuda:0)
target_device = utonia_device
for k, v in list(item.items()):
    if isinstance(v, torch.Tensor):
        item[k] = v.to(target_device, non_blocking=True)

attach_utonia_feature(item, utonia_model, utonia_device)
print("attach_utonia_feature done. utonia_feat:", item["utonia_feat"].shape)
assert item["utonia_feat"].shape[1] == 1224, item["utonia_feat"].shape
print("Final check OK: utonia_feat is [N, 1224] as expected by model_token.py")
