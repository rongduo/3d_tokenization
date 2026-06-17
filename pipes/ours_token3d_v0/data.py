"""
训练时候，加载代码验证 ： 
mkdir -p /x2robot_v2/lanejin/new_data/cosmo3d/results/featscheck && python -m pipes.ours_token3d.data --data_root /x2robot_v2/lanejin/new_data/cosmo3d/dataset/datasets/d3compat --index 0 --debug_out_dir /x2robot_v2/lanejin/new_data/cosmo3d/results/featscheck
"""



# augmentation code builds upon https://huggingface.co/Pointcept/PointTransformerV3/blob/main/s3dis-semseg-pt-v3m1-1-ppt-extreme/config.py
from model.data.augmentation import *
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from collections.abc import Mapping, Sequence
import argparse
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import os
import open3d as o3d
import h5py
import json
import glob
import sys
import importlib.util
from common.utils import rotate_pts

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
DEFAULT_SIGLIP_LOCAL_DIR = "/x2robot_v2/lanejin/new_data/largemodelcahe/siglip-base-patch16-224"
DEFAULT_UTONIA_CKPT_DIR = "/x2robot_v2/lanejin/new_data/largemodelcahe/utonia/ckpt"


def load_siglip_model_and_tokenizer():
    """
    Load SigLIP with a local-first strategy:
    1) If SIGLIP_LOCAL_DIR exists, load from local only.
    2) Otherwise, download from Hugging Face (optionally using SIGLIP_CACHE_DIR).
    """
    local_dir = os.environ.get("SIGLIP_LOCAL_DIR", DEFAULT_SIGLIP_LOCAL_DIR).strip()
    cache_dir = os.environ.get("SIGLIP_CACHE_DIR", "").strip()
    cache_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    if local_dir:
        if os.path.isdir(local_dir):
            model = AutoModel.from_pretrained(local_dir, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
            return model, tokenizer
        print(f"[siglip] SIGLIP_LOCAL_DIR does not exist: {local_dir}; fallback to remote download.")

    model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, **cache_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID, **cache_kwargs)
    return model, tokenizer


def ensure_utonia_local_ckpt() -> None:
    """
    Ensure Utonia default cache path has local checkpoint first.
    """
    local_ckpt_dir = os.environ.get("UTONIA_CKPT_DIR", DEFAULT_UTONIA_CKPT_DIR).strip()
    local_ckpt = os.path.join(local_ckpt_dir, "utonia.pth")
    if not os.path.isfile(local_ckpt):
        print(f"[utonia] local ckpt not found: {local_ckpt}; fallback to online download.")
        return

    default_cache_dir = os.path.expanduser("~/.cache/utonia/ckpt")
    default_ckpt = os.path.join(default_cache_dir, "utonia.pth")
    if os.path.isfile(default_ckpt):
        return

    os.makedirs(default_cache_dir, exist_ok=True)
    try:
        os.symlink(local_ckpt, default_ckpt)
    except FileExistsError:
        pass
    except OSError:
        # Symlink may fail on some mounts/filesystems; copy as fallback.
        import shutil
        shutil.copy2(local_ckpt, default_ckpt)


def resolve_utonia_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = os.environ.get("LOCAL_RANK", "").strip()
    if local_rank.isdigit():
        return f"cuda:{int(local_rank)}"
    rank = os.environ.get("RANK", "").strip()
    if rank.isdigit():
        return f"cuda:{int(rank)}"
    return f"cuda:{torch.cuda.current_device()}"


def rotate_point_cloud_with_matrix(points, matrix):
    """使用给定的旋转矩阵旋转点云（纯Tensor实现）"""
    # 确保输入为Tensor（若输入是numpy会自动转换，保持原函数兼容）
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32, device=points.device)
    
    # 核心旋转操作：点云矩阵与旋转矩阵的转置相乘
    # points形状：[B, N, 3] 或 [N, 3]；matrix形状：[3, 3]
    # 矩阵乘法使用torch.matmul，自动适配批次维度
    rotated_points = torch.matmul(points, matrix.T)
    
    return rotated_points

def prep_points_train(xyz, rgb, normal, mask2pt, canoncial_color=None, tokens=None):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame x revert, y z shift
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {
        "coord": xyz_change_axis,
        "color": rgb,
        "normal": normal,
        "mask2pt": mask2pt,
        "canoncial_color": canoncial_color,
    }
    if tokens is not None:
        data_dict["tokens"] = tokens
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = RandomScale(scale=[0.8, 1.2], anisotropic=True)(data_dict)  # size 增强
    data_dict = RandomRotate(angle=[-1, 1],axis='z',center=[0, 0, 0],p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='y',p=1)(data_dict)
    # data_dict = RandomScale(scale=[0.9, 1.1])(data_dict)
    # data_dict = RandomScale(scale=[0.8, 1.2], anisotropic=True)(data_dict) (不对，应该先尺寸增强，再旋转才行 ‘ 因为旋转之前在canonical space)
    data_dict = RandomFlip(p=0.5)(data_dict)
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)
    data_dict = ChromaticAutoContrast(p=0.2,blend_factor=None)(data_dict)
    data_dict = ChromaticTranslation(p=0.95, ratio=0.05)(data_dict)
    data_dict = ChromaticJitter(p=0.95, std=0.05)(data_dict)
    gridsample_keys = ["coord", "color", "normal"]
    if tokens is not None:
        gridsample_keys.append("tokens")
    data_dict = GridSample(
        grid_size=0.02,
        hash_type='fnv',
        mode='train',
        keys=tuple(gridsample_keys),
        return_grid_coord=True,
    )(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    if canoncial_color is not None:
        collect_keys = ["coord", "grid_coord", "mask2pt", "canoncial_color"]
        if tokens is not None:
            collect_keys.append("tokens")
        data_dict = Collect(
            keys=tuple(collect_keys),
            offset_keys_dict={"offset": "coord", "mask_offset": "mask2pt"},
            feat_keys=("color", "normal"),
        )(data_dict)
        return data_dict
        
    collect_keys = ["coord", "grid_coord", "mask2pt"]
    if tokens is not None:
        collect_keys.append("tokens")
    data_dict = Collect(
        keys=tuple(collect_keys),
        offset_keys_dict={"offset": "coord", "mask_offset": "mask2pt"},
        feat_keys=("color", "normal"),
    )(data_dict)
    return data_dict



def prep_points_val(xyz, rgb, normal, mask2pt, pt2face):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal, "mask2pt": mask2pt, 'point2face': pt2face}
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict) # mode train is used in original code, text will subsample points n times and create many samples out of one sample
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord', 'mask2pt', 'point2face'),
                        feat_keys=('color', 'normal'))(data_dict)
    return data_dict

def prep_points_val3d(xyz, rgb, normal, gt, xyz_full, gt_full):
    # the input xyz is expected to be ~5000 points, and the returned coord will be grid-sampled to e.g. 3000
    # the xyz_full can be however dense, e.g. 300k points for partnete, gt_full is the same size as xyz_full
    # but for sparser point clouds we can keep them the same
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    xyz_full_change_axis = np.concatenate([-xyz_full[:,0].reshape(-1,1), xyz_full[:,2].reshape(-1,1), xyz_full[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal, "gt":gt, "xyz_full": xyz_full_change_axis}
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict) # mode train is used in original code, text will subsample points n times and create many samples out of one sample
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord', "gt", "xyz_full"),
                        feat_keys=('color', 'normal'))(data_dict)
    data_dict["gt_full"] = gt_full
    return data_dict


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        if len(batch)>1:
            try:
                all_cat = torch.cat(list(batch))
                return all_cat
            except Exception:
                return list(batch) # not uniform shape
        else: # only one item, e.g. mask2pt, return itself
            return batch[0]
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        if isinstance(batch[0][0], str):
            return batch
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch_new = {key: collate_fn([d[key] for d in batch]) for key in batch[0] if key != "mask2pt"}
        if "mask2pt" in batch[0]:
            collated_mask2pt =  [d["mask2pt"] for d in batch]
            batch_new["mask2pt"] = collated_mask2pt
        for key in batch_new.keys():
            if "offset" in key:
                batch_new[key] = torch.cumsum(batch_new[key], dim=0)
        return batch_new
    else:
        return default_collate(batch)

def get_shapenetp_prompts(cat):
        with open('evaluation/benchmark/benchmark_reproducibility/shapenetpart/topk_prompts.json') as f:
            all_prompts = json.load(f)
        return all_prompts[cat]

def normalize_to_rgb(points):
    """
    将点云坐标归一化到0-255范围，作为RGB颜色值（纯Tensor实现）
    """
    # 确保输入是torch.Tensor
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    
    # 找到每个维度的最小值和最大值
    min_vals, _ = torch.min(points, dim=0)
    max_vals, _ = torch.max(points, dim=0)
    
    # 计算范围，添加微小值避免除零
    ranges = max_vals - min_vals
    # 处理恒值维度，使用torch.where替代numpy的布尔索引
    ranges = torch.where(ranges < 1e-8, torch.tensor(1e-8, device=points.device), ranges)
    
    # 归一化到0-1范围，再转换到0-255并转为整数
    normalized = (points - min_vals) / ranges
    # 确保值在[0, 1]范围内，防止由于浮点计算误差导致的微小溢出
    rgb = torch.clamp(normalized, 0.0, 1.0)  # 输出是0-1
    # rgb = (rgb * 255).to(torch.uint8)  # 输出是0-255
    
    return rgb


# ===================== UTONIA DEBUG BLOCK (easy to remove) =====================
def _pca_color_from_feat(feat: torch.Tensor, brightness: float = 1.2) -> torch.Tensor:
    if feat.numel() == 0 or feat.shape[1] < 3:
        return torch.zeros((feat.shape[0], 3), dtype=torch.float32)
    q = min(9, feat.shape[1])
    _, _, v = torch.pca_lowrank(feat, center=True, niter=5, q=q)
    proj = feat @ v
    if proj.shape[1] >= 6:
        proj3 = proj[:, :3] * 0.6 + proj[:, 3:6] * 0.4
    else:
        proj3 = proj[:, :3]
    pmin = proj3.min(dim=0, keepdim=True)[0]
    pmax = proj3.max(dim=0, keepdim=True)[0]
    div = torch.clamp(pmax - pmin, min=1e-6)
    color = (proj3 - pmin) / div * float(brightness)
    return color.clamp(0.0, 1.0)


def _write_ply(coord: torch.Tensor, color: torch.Tensor, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.detach().cpu().numpy().astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(color.detach().cpu().numpy().astype(np.float64))
    o3d.io.write_point_cloud(save_path, pcd)
# ===================== END UTONIA DEBUG BLOCK =====================


class TrainingData(Dataset):
    def __init__(self, data_root):
        self.obj_path_list = []
        self.data_root = data_root
        # with open(f"{data_root}/split/train.txt", "r") as f:
        #     self.obj_path_list = f.read().splitlines()
        with open(f"{data_root}/train.txt", "r") as f:
            self.obj_path_list = f.read().splitlines()

        self.model, self.tokenizer = load_siglip_model_and_tokenizer()  # .cuda()


        # Utonia init for point-level feature extraction.
        self.utonia_model = None
        self.utonia_device = "cpu"
        self.utonia_enabled = False
        utonia_root = os.environ.get("UTONIA_ROOT", "/x2robot_v2/lanejin/new_data/Utonia")
        if utonia_root not in sys.path:
            sys.path.insert(0, utonia_root)
        copule_path = os.path.join(utonia_root, "pipes", "exfeats", "feature_pca_copule.py")
        spec = importlib.util.spec_from_file_location("utonia_feature_pca_copule", copule_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module from: {copule_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        init_utonia_model = mod.init_utonia_model

        ensure_utonia_local_ckpt()
        self.utonia_model, self.utonia_device = init_utonia_model(device=resolve_utonia_device())
        self.utonia_enabled = True
        # ===================== UTONIA DEBUG BLOCK (easy to remove) =====================
        self.utonia_debug = os.environ.get("UTONIA_DEBUG", "0") == "1"
        self.utonia_debug_out_dir = os.environ.get(
            "UTONIA_DEBUG_OUT_DIR",
            "/x2robot_v2/lanejin/new_data/cosmo3d/results/utonia_debug",
        )
        self.utonia_debug_max_items = int(os.environ.get("UTONIA_DEBUG_MAX_ITEMS", "1"))
        self._utonia_debug_count = 0
        # ===================== END UTONIA DEBUG BLOCK =====================
 


    def process_single_point_cloud(self, pc_tensor):
        """
        对单个点云torch张量执行中心化+单位化处理（等价于numpy版本，支持GPU/CPU，兼容Autograd）
        :param pc_tensor: 输入[N, 3]格式的点云torch张量（float32/float64）
        :return: 处理后的[N, 3]格式点云torch张量（与输入张量类型、设备一致）
        """
        # 步骤1：计算点云几何参数（使用torch内置函数，替代numpy的min/max）
        # dim=0 表示沿着样本数量维度（N）计算，得到每个坐标维度（x/y/z）的最值
        pc_min, _ = torch.min(pc_tensor, dim=0)  # 返回 [3,] 张量（x_min, y_min, z_min）
        pc_max, _ = torch.max(pc_tensor, dim=0)  # 返回 [3,] 张量（x_max, y_max, z_max）
        
        # 中心点（用于中心化，torch张量运算，保持设备和类型一致）
        center = (pc_max + pc_min) / 2.0  # [3,] 张量
        
        # 各维度值域和全局最大值域（用于单位化，保持形状）
        xyz_range = pc_max - pc_min  # [3,] 张量（x_range, y_range, z_range）
        global_max_range = torch.max(xyz_range)  # 标量张量，全局最大值域
        
        # 步骤2：中心化（平移到原点，广播机制，与numpy等价）
        pc_centered = pc_tensor - center  # [N, 3] 张量，自动广播center到[N, 3]
        
        # 步骤3：全局单位化（鲁棒性处理，避免除以0，torch.where替代条件判断更优雅）
        # 安全值域：如果global_max_range为0，用1e-8替代，否则用自身
        safe_max_range = torch.where(
            global_max_range == 0,
            torch.tensor(1e-8, dtype=pc_tensor.dtype, device=pc_tensor.device),
            global_max_range
        )
        pc_normalized = pc_centered / safe_max_range
        
        return pc_normalized

    def __getitem__(self, item):
        data_dir = self.obj_path_list[item]
        # 所有数据加载均在CPU完成，不调用.cuda()
        with open(f"{data_dir}/mask_labels.txt", "r") as f:
            labels = f.read().splitlines()
        mask_pts = torch.load(f"{data_dir}/mask2points.pt", weights_only=True).cpu()  # 强制CPU
        pts_xyz = torch.load(f"{data_dir}/points.pt", weights_only=True).cpu()
        pts_xyz = self.process_single_point_cloud(pts_xyz)  # 中心化+单位化
        normal = torch.load(f"{data_dir}/normals.pt", weights_only=True).cpu()
        pts_rgb = torch.load(f"{data_dir}/rgb.pt", weights_only=True).cpu() * 255

        # 加载part 标签 ； 这个还没经过下采样呢，别忘了
        tokens = torch.load(f"{data_dir}/parts.pt", weights_only=True).cpu()


        # NOTE: canonical-space rotation is disabled in current experiment.

        # 获得规范空间的颜色
        canoncial_color = normalize_to_rgb(pts_xyz)

        point_dict = prep_points_train(
            pts_xyz.numpy(),
            pts_rgb.numpy(),
            normal.numpy(),
            mask_pts.numpy(),
            canoncial_color.numpy(),
            tokens=tokens.numpy(),
        )

        '''### token验证
        # ===================== TOKEN DEBUG BLOCK (easy to remove) =====================
        if self.utonia_debug and self._utonia_debug_count < self.utonia_debug_max_items:
            if "tokens" not in point_dict:
                raise RuntimeError(f"tokens missing after prep_points_train for {data_dir}")
            token_coord = point_dict["coord"].detach().cpu().float()
            token_vals = point_dict["tokens"].detach().cpu().reshape(-1).long()
            if token_coord.shape[0] != token_vals.shape[0]:
                raise RuntimeError(
                    f"token/coord count mismatch for {data_dir}: "
                    f"coord={token_coord.shape[0]}, token={token_vals.shape[0]}"
                )

            unique_tokens = torch.unique(token_vals, sorted=True)
            token_rgb = torch.zeros((token_vals.shape[0], 3), dtype=torch.float32)
            token2color = {}
            for tk in unique_tokens.tolist():
                # deterministic pseudo-color by token id
                r = ((tk * 73 + 29) % 255) / 255.0
                g = ((tk * 131 + 67) % 255) / 255.0
                b = ((tk * 197 + 101) % 255) / 255.0
                c = torch.tensor([r, g, b], dtype=torch.float32)
                token_rgb[token_vals == tk] = c
                token2color[int(tk)] = (float(r), float(g), float(b))

            safe_name = os.path.basename(data_dir).replace("/", "_")
            _write_ply(
                token_coord,
                token_rgb,
                os.path.join(self.utonia_debug_out_dir, f"{safe_name}_token_color.ply"),
            )
            map_path = os.path.join(self.utonia_debug_out_dir, f"{safe_name}_token_color_map.txt")
            with open(map_path, "w", encoding="utf-8") as f:
                f.write("token_id\tr\tg\tb\n")
                for tk in unique_tokens.tolist():
                    r, g, b = token2color[int(tk)]
                    f.write(f"{int(tk)}\t{r:.6f}\t{g:.6f}\t{b:.6f}\n")
        # ===================== END TOKEN DEBUG BLOCK ====================='''

        # 获得utonia的特征：直接使用 prep_points_train 后的点（避免再次 GridSample 降采样）
        n_pts_after_prep = int(point_dict["coord"].shape[0]) if "coord" in point_dict else 0
        utonia_feat = torch.empty((n_pts_after_prep, 0), dtype=torch.float32)
        utonia_coord = point_dict["coord"].detach().cpu().float() if "coord" in point_dict else torch.empty((0, 3), dtype=torch.float32)
        if self.utonia_enabled and self.utonia_model is not None:
            # Utonia stem expects 9-dim input feature: [coord(3), color(3), normal(3)].
            utonia_coord_in = point_dict["coord"]
            utonia_feat_in = torch.cat([utonia_coord_in, point_dict["feat"]], dim=1)
            utonia_input = {
                "coord": utonia_coord_in,
                "grid_coord": point_dict["grid_coord"],
                "offset": point_dict["offset"],
                "feat": utonia_feat_in,
            }
            if utonia_input["feat"].ndim != 2 or int(utonia_input["feat"].shape[1]) != 9:
                raise RuntimeError(
                    f"Utonia input feat dim must be 9, got {tuple(utonia_input['feat'].shape)}"
                )
            with torch.inference_mode():
                for key in list(utonia_input.keys()):
                    if isinstance(utonia_input[key], torch.Tensor):
                        use_cuda = str(self.utonia_device).startswith("cuda")
                        utonia_input[key] = utonia_input[key].to(
                            self.utonia_device,
                            non_blocking=use_cuda,
                        )
                utonia_out = self.utonia_model(utonia_input)
                # Keep identical to Utonia featurize path: propagate to dense level.
                for _ in range(2):
                    assert "pooling_parent" in utonia_out.keys()
                    assert "pooling_inverse" in utonia_out.keys()
                    parent = utonia_out.pop("pooling_parent")
                    inverse = utonia_out.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, utonia_out.feat[inverse]], dim=-1)
                    utonia_out = parent
                while "pooling_parent" in utonia_out.keys():
                    assert "pooling_inverse" in utonia_out.keys()
                    parent = utonia_out.pop("pooling_parent")
                    inverse = utonia_out.pop("pooling_inverse")
                    parent.feat = utonia_out.feat[inverse]
                    utonia_out = parent
                utonia_feat = utonia_out.feat.detach().cpu().float()
                utonia_coord = utonia_out.coord.detach().cpu().float()
            if utonia_feat.ndim != 2 or utonia_feat.shape[1] == 0:
                raise RuntimeError(
                    f"Utonia returned invalid feature shape for {data_dir}: {tuple(utonia_feat.shape)}"
                )

        point_dict["utonia_feat"] = utonia_feat
        point_dict["utonia_coord"] = utonia_coord
        '''# ===================== UTONIA DEBUG BLOCK (easy to remove) =====================
        if self.utonia_debug and self._utonia_debug_count < self.utonia_debug_max_items:
            sampled_coord = point_dict["coord"].detach().cpu().float()
            sampled_cnt = int(sampled_coord.shape[0])
            out_cnt = int(utonia_coord.shape[0])
            cnt_equal = sampled_cnt == out_cnt
            allclose = False
            max_diff = None
            if cnt_equal:
                allclose = bool(torch.allclose(utonia_coord, sampled_coord, atol=1e-6, rtol=1e-5))
                max_diff = float((utonia_coord - sampled_coord).abs().max().item())

            print(f"[UTONIA_DEBUG] sampled_count={sampled_cnt} out_count={out_cnt} count_equal={cnt_equal}")
            if max_diff is not None:
                print(f"[UTONIA_DEBUG] coord_allclose={allclose} max_abs_diff={max_diff:.6e}")
            else:
                print("[UTONIA_DEBUG] coord_allclose=skipped (count mismatch)")

            safe_name = os.path.basename(data_dir).replace("/", "_")
            pca_color = _pca_color_from_feat(utonia_feat)
            _write_ply(
                utonia_coord,
                pca_color,
                os.path.join(self.utonia_debug_out_dir, f"{safe_name}_utonia_feat.ply"),
            )
            sampled_color = normalize_to_rgb(sampled_coord)
            _write_ply(
                sampled_coord,
                sampled_color,
                os.path.join(self.utonia_debug_out_dir, f"{safe_name}_sampled_coord.ply"),
            )
            self._utonia_debug_count += 1
        # ===================== END UTONIA DEBUG BLOCK ====================='''

        point_dict['labels'] = labels

        ## 文本编码仅在CPU处理，不转移到GPU
        inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
        # 移除所有.cuda()操作，保持在CPU
        with torch.no_grad():
            # 若self.model在GPU，此处会报错！需确保文本模型暂时在CPU
            text_feat = self.model.get_text_features(** inputs)  # 此时在CPU
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        point_dict['label_embeds'] = text_feat  # 保持CPU张量
        return point_dict
    
    def __len__(self):
        return len(self.obj_path_list)


class EvalData(Dataset):
    def __init__(self, data_root, split):
        assert split in ["val", "test", "train"]
        self.data_root = data_root
        self.obj_path_list = []
        with open(f"{data_root}/{split}.txt", "r") as f:
            self.obj_path_list = f.read().splitlines()
        self.model, self.tokenizer = load_siglip_model_and_tokenizer()
        self.model = self.model.cuda()
    
    def __getitem__(self, item):
        return_dict = {}
        # name_uid = self.obj_path_list[item]
        # file_path = f"{self.data_root}/rendered/{name_uid}/oriented"
        # uid = name_uid # name_uid.split("_")[-1]
        # with open(f"{file_path}/masks/merged/mask_labels.txt", "r") as f:
        #     labels = f.read().splitlines()
        # masks = torch.load(f"{file_path}/masks/merged/allmasks.pt")
        # mask_view_idxs = torch.load(f"{file_path}/masks/merged/mask2view.pt")
        # pt2face = torch.load(f"{self.data_root}/points/{uid}/point2face.pt")
        # pix2face = torch.load(f"{file_path}/pix2face.pt")
        # pts_xyz = torch.load(f"{self.data_root}/points/{uid}/points.pt").cpu()
        # normal = torch.load(f"{self.data_root}/points/{uid}/normals.pt").cpu()
        # pts_rgb = torch.load(f"{self.data_root}/points/{uid}/rgb.pt").cpu()*255
        # mask_pts = torch.load(f"{file_path}/masks/merged/mask2points.pt").cpu()

        # return_dict = prep_points_val(pts_xyz[0], pts_rgb[0], normal[0], mask_pts, pt2face)
        

        ## encode label
        inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs) # n_masks, feat_dim (768)
        
        #normalize
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        return_dict['label_embeds'] = text_feat # n_cur_mask, dim_feat, need to be padded
        return_dict['masks'] = masks
        return_dict['mask_view_idxs'] = mask_view_idxs
        return_dict['pixel2face'] = pix2face
        return_dict['labels'] = labels

        return return_dict
    
    def __len__(self):
        return len(self.obj_path_list)
    

class EvalData3D(Dataset):
    def __init__(self, split, root, decorated=True, use_tuned_prompt=False, visualization=False):
        assert split in ["seenclass", "unseen", "shapenetpart"]
        
        class_uids = sorted(os.listdir(f"{root}/{split}"))
        self.split = split
        self.decorated = decorated
        self.use_tuned_promopt = use_tuned_prompt
        self.obj_path_list = [f"{root}/{split}/{class_uid}" for class_uid in class_uids]
        self.visualization = visualization
        self.model, self.tokenizer = load_siglip_model_and_tokenizer()  # dim 768
    
    def __getitem__(self, item):
        return_dict = {}
        file_path = self.obj_path_list[item]
        classname = " ".join(file_path.split("/")[-1].split("_")[:-1])
        pcd = o3d.io.read_point_cloud(f"{file_path}/points5000.pcd")
        with open(f"{file_path}/label_map.json") as f:
            label_dict = json.load(f)
        ordered_label_list = []
        for i in range(len(label_dict)):
            ordered_label_list.append(label_dict[str(i+1)])
        
        pts_xyz = torch.tensor(np.asarray(pcd.points)).float()
        normal = torch.tensor(np.asarray(pcd.normals))
        pts_rgb = torch.tensor(np.asarray(pcd.colors))*255

        gt = torch.tensor(np.load(f"{file_path}/labels.npy"))

        return_dict = prep_points_val3d(pts_xyz, pts_rgb, normal, gt, pts_xyz, gt)

        ## encode label
        if self.use_tuned_promopt and self.split == "shapenetpart":
            ordered_label_list = get_shapenetp_prompts(classname)
        elif self.decorated:
            ordered_label_list = [f"{part} of a {classname}" for part in ordered_label_list]
        if self.visualization:
            print(ordered_label_list)
        inputs = self.tokenizer(ordered_label_list, padding="max_length", return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            text_feat = self.model.cuda().get_text_features(**inputs) # n_masks, feat_dim (768)
        
        #normalize
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        return_dict['label_embeds'] = text_feat
        return_dict['class_name'] = classname
        return_dict['file_path'] = file_path
        return_dict['xyz_visualization'] = pts_xyz # this is only for visualization, not interpolation bc interpolation happens after scaling

        return return_dict
    
    def __len__(self):
        return len(self.obj_path_list)
    

class EvalPartNetE(Dataset):
    def __init__(self, data_root, category, apply_rotation=False, subset=False, decorated=True):
        
        ids = sorted(os.listdir(f"{data_root}/test/{category}"))
        if subset:
            with open('evaluation/benchmark/benchmark_reproducibility/partnete/subset_idxs.json', 'r') as file:
                subset_idxs= json.load(file)[category]
                self.obj_path_list = [f"{data_root}/test/{category}/{id}" for id in subset_idxs]
        else:
            self.obj_path_list = [f"{data_root}/test/{category}/{id}" for id in ids if "txt" not in id]
        

        self.category = category
        self.decorated = decorated
        self.apply_rotation = apply_rotation
        with open(f"{data_root}/PartNetE_meta.json") as f:
            all_mapping = json.load(f)
        self.part_names = all_mapping[category]
        if self.decorated:
            self.part_names = [f"{part} of a {category}" for part in self.part_names]
        

        # misc.
        self.model, self.tokenizer = load_siglip_model_and_tokenizer()

        ## encode label
        inputs = self.tokenizer(self.part_names, padding="max_length", return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            text_feat = self.model.cuda().get_text_features(**inputs) # n_masks, feat_dim (768)
        
        #normalize
        self.text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    
 
    def __getitem__(self, item):
        return_dict = {}
        file_path = self.obj_path_list[item]
        pcd = o3d.io.read_point_cloud(f"{file_path}/pc.ply")
        rot = torch.load(f"{file_path}/rand_rotation.pt")
        
        pts_xyz = torch.tensor(np.asarray(pcd.points)).float()
        pts_rgb = torch.tensor(np.asarray(pcd.colors))*255
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=300))
        normal = torch.tensor(np.asarray(pcd.normals)).float()

        if self.apply_rotation:
            pts_xyz = rotate_pts(pts_xyz, rot)
            normal = rotate_pts(normal, rot)

        # normalize
        # this is the same preprocessing done before training
        center = pts_xyz.mean(0)
        scale = max((pts_xyz - center).abs().max(0)[0])
        pts_xyz -= center
        pts_xyz *= (0.75 / float(scale)) # put in 1.5-size box
        
        # subsample 5000 pts
        random_indices = torch.randint(0, pts_xyz.shape[0], (5000,))
        pts_xyz_subsampled = pts_xyz[random_indices]
        pts_rgb_subsampled = pts_rgb[random_indices]
        normal_subsampled = normal[random_indices]
        
        gt = torch.tensor(np.load(f"{file_path}/label.npy",allow_pickle=True).item()['semantic_seg'])+1 # we make it agree with objaverse, 0 is unlabeled and 1-k labeled
        gt_subsampled = gt[random_indices]

        return_dict = prep_points_val3d(pts_xyz_subsampled, pts_rgb_subsampled, normal_subsampled, gt_subsampled, pts_xyz, gt) # we also pass in the dense point clouds

        return_dict['label_embeds'] = self.text_feat
        return_dict['class_name'] = self.category
        return_dict["xyz_visualization"] = torch.tensor(np.asarray(pcd.points)).float()

        return return_dict
    
    def __len__(self):
        return len(self.obj_path_list)
    

class EvalShapeNetPart(Dataset):
    def __init__(self, data_path, class_choice, apply_rotation=False, subset=False, decorated=True, use_tuned_prompt=False):
        self.apply_rotation = apply_rotation
        self.decorated = decorated
        file = glob.glob(os.path.join(data_path, 'hdf5_data', '*test*.h5'))
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            all_seg.append(seg)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        all_seg = np.concatenate(all_seg, axis=0)
        all_rotation = torch.load(f"evaluation/benchmark/benchmark_reproducibility/shapenetpart/random_rotation_test.pt")

        self.data = all_data
        self.label = all_label
        self.seg = all_seg
        self.all_rotation = all_rotation

        # misc.
        self.model, self.tokenizer = load_siglip_model_and_tokenizer()

        self.cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['base','lampshade', 'fixing bracket', 'stem'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
        
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

        id_choice = self.cat2id[class_choice]
        self.class_choice = class_choice
        indices = (self.label == id_choice).squeeze()
        self.data = self.data[indices]
        self.label = self.label[indices]
        self.seg = self.seg[indices]
        self.all_rotation = self.all_rotation[indices]
        self.seg_start_index = self.index_start[id_choice]

        ## encode label
        if use_tuned_prompt:
            part_names = get_shapenetp_prompts(class_choice)
        elif self.decorated:
            part_names = [f"{part} of a {class_choice}" for part in self.cat2part[class_choice]]
        else:
            part_names = self.cat2part[class_choice]

        inputs = self.tokenizer(part_names, padding="max_length", return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            text_feat = self.model.cuda().get_text_features(**inputs) # n_masks, feat_dim (768)
        
        #normalize
        self.text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        if subset:
            # get subset
            with open('evaluation/benchmark/benchmark_reproducibility/shapenetpart/subset_idxs.json', 'r') as file:
                subsets= json.load(file)
            subset_idxs = [int(ind) for ind in subsets[class_choice]]
            self.data = self.data[subset_idxs]
            self.label = self.label[subset_idxs]
            self.seg = self.seg[subset_idxs]
            self.all_rotation = self.all_rotation[subset_idxs]

    def __getitem__(self, item):
        pointcloud = self.data[item]
        cat = self.class_choice
        gt = self.seg[item]- self.index_start[self.cat2id[cat]] + 1
        rot = self.all_rotation[item,:]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100))

        pts_xyz = torch.tensor(pointcloud).float()
        pts_rgb = torch.ones(pts_xyz.shape)*127.5 # no color  为什么这里不用颜色了呢
        normal = torch.tensor(np.asarray(pcd.normals)).float()

        if self.apply_rotation:
            pts_xyz = rotate_pts(pts_xyz, rot)
            normal = rotate_pts(normal, rot)

        xyz_visualize = pts_xyz.clone()
        
        return_dict = prep_points_val3d(pts_xyz, pts_rgb, normal, gt, pts_xyz, gt)

        return_dict['label_embeds'] = self.text_feat # n_cur_mask, dim_feat, need to be padded
        return_dict['class_name'] = cat
        return_dict["xyz_visualization"] = xyz_visualize

        return return_dict
    
    def __len__(self):
        return self.data.shape[0]


# ===================== UTONIA DEBUG BLOCK (easy to remove) =====================
def _main_single_example_test():
    parser = argparse.ArgumentParser(description="Single example test for TrainingData + Utonia feature checks.")
    parser.add_argument(
        "--data_root",
        default="/x2robot_v2/lanejin/new_data/cosmo3d/dataset/datasets/datasets/train/3dcompat/forfind3dtrain",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--debug_out_dir",
        default="/x2robot_v2/lanejin/new_data/cosmo3d/results/utonia_debug",
    )
    args = parser.parse_args()

    os.environ["UTONIA_DEBUG"] = "1"
    os.environ["UTONIA_DEBUG_MAX_ITEMS"] = "1"
    os.environ["UTONIA_DEBUG_OUT_DIR"] = args.debug_out_dir

    dataset = TrainingData(args.data_root)
    idx = max(0, min(args.index, len(dataset) - 1))
    sample = dataset[idx]
    sampled_coord = sample["coord"].detach().cpu().float()
    utonia_coord = sample["utonia_coord"].detach().cpu().float()
    utonia_feat = sample["utonia_feat"].detach().cpu().float()

    print(f"[MAIN_TEST] sample_index={idx}")
    print(f"[MAIN_TEST] sampled_count={sampled_coord.shape[0]}")
    print(f"[MAIN_TEST] utonia_coord_count={utonia_coord.shape[0]}")
    print(f"[MAIN_TEST] utonia_feat_shape={tuple(utonia_feat.shape)}")
    if sampled_coord.shape[0] == utonia_coord.shape[0]:
        eq = torch.allclose(utonia_coord, sampled_coord, atol=1e-6, rtol=1e-5)
        md = (utonia_coord - sampled_coord).abs().max().item()
        print(f"[MAIN_TEST] coord_allclose={bool(eq)} max_abs_diff={md:.6e}")
    else:
        print("[MAIN_TEST] coord_allclose=skipped (count mismatch)")
    print(f"[MAIN_TEST] debug_ply_dir={args.debug_out_dir}")


if __name__ == "__main__":
    _main_single_example_test()
# ===================== END UTONIA DEBUG BLOCK =====================
