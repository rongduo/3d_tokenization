# augmentation code builds upon https://huggingface.co/Pointcept/PointTransformerV3/blob/main/s3dis-semseg-pt-v3m1-1-ppt-extreme/config.py
from model.data.augmentation import *
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import os
import open3d as o3d
import h5py
import json
import glob
from common.utils import rotate_pts

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

def prep_points_train(xyz, rgb, normal, mask2pt):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame x revert, y z shift
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal, "mask2pt": mask2pt}
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
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord', 'mask2pt'),
                        offset_keys_dict={"offset":"coord", "mask_offset":"mask2pt"},
                        feat_keys=('color', 'normal'))(data_dict)
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


class TrainingData(Dataset):
    def __init__(self, data_root):
        self.obj_path_list = []
        self.data_root = data_root
        # with open(f"{data_root}/split/train.txt", "r") as f:
        #     self.obj_path_list = f.read().splitlines()
        with open(f"{data_root}/train.txt", "r") as f:
            self.obj_path_list = f.read().splitlines()

        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")  # .cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")


        ### 读取规范标签
        cate_rots_json = '/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/crosscatesalign/category_alignment_rot_dict.json'
        with open(cate_rots_json, 'r') as f:
            self.cate_rots_dict = json.load(f)
        shape2cates_json = '/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/train_object_category_mapping.json'
        with open(shape2cates_json, 'r') as f:
            self.shape2cates_dict = json.load(f)
 

    

    def __getitem__(self, item):
        data_dir = self.obj_path_list[item]
        # 所有数据加载均在CPU完成，不调用.cuda()
        with open(f"{data_dir}/mask_labels.txt", "r") as f:
            labels = f.read().splitlines()
        mask_pts = torch.load(f"{data_dir}/mask2points.pt").cpu()  # 强制CPU
        pts_xyz = torch.load(f"{data_dir}/points.pt").cpu()
        normal = torch.load(f"{data_dir}/normals.pt").cpu()
        pts_rgb = torch.load(f"{data_dir}/rgb.pt").cpu() * 255

        ### 对点云和法向量进行旋转，对齐到规范空间中
        obj_id = data_dir.split("'")[1]
        obj_id_str = obj_id.decode('utf-8') if isinstance(obj_id, bytes) else str(obj_id)
        cate = self.shape2cates_dict[obj_id_str]
        # 判断cate中是否有 "_" 如果有 换成 空格
        if "_" in cate:
            cate = cate.replace("_", " ")
        rot = torch.tensor(self.cate_rots_dict[cate])
        pts_xyz = rotate_point_cloud_with_matrix(pts_xyz, rot)
        normal = rotate_point_cloud_with_matrix(normal, rot)

        
        point_dict = prep_points_train(pts_xyz.numpy(), pts_rgb.numpy(), normal.numpy(), mask_pts.numpy())

        point_dict['labels'] = labels

        text_feat_path = f"{data_dir}/text_feat.pt"  # 缓存路径与数据文件同目录
        if os.path.exists(text_feat_path):
            # 存在缓存，直接加载（保持在CPU，避免子进程GPU操作）
            text_feat = torch.load(text_feat_path, map_location="cpu")
        else:
            ## 文本编码仅在CPU处理，不转移到GPU
            inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
            # 移除所有.cuda()操作，保持在CPU
            with torch.no_grad():
                # 若self.model在GPU，此处会报错！需确保文本模型暂时在CPU
                text_feat = self.model.get_text_features(** inputs)  # 此时在CPU
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
            # 保存到缓存路径，供下次加载
            torch.save(text_feat, text_feat_path)

        point_dict['label_embeds'] = text_feat  # 保持CPU张量
        return point_dict
    
    def __len__(self):
        return len(self.obj_path_list)

    '''def __getitem__(self, item):
        data_dir = self.obj_path_list[item]

        # # 更换为私有盘中的数据集 将 /apdcephfs_cq11/share_303570626/lanejin/dataset  换为   /apdcephfs/share_303570626/lanejin/dataset/
        # old_path = "/apdcephfs_cq11/share_303570626/lanejin/dataset"
        # new_path = "/apdcephfs/share_303570626/lanejin/dataset/"
        # data_dir = data_dir.replace(old_path, new_path)

        with open(f"{data_dir}/mask_labels.txt", "r") as f:
            labels = f.read().splitlines()
        mask_pts = torch.load(f"{data_dir}/mask2points.pt").cpu()
        pts_xyz = torch.load(f"{data_dir}/points.pt").cpu()
        normal = torch.load(f"{data_dir}/normals.pt").cpu()
        pts_rgb = torch.load(f"{data_dir}/rgb.pt").cpu()*255

        # debug ； 将重复的标签进行合并
        queries_np = np.array(labels)  # 1. 建立标签到索引的映射
        unique_labels, label_ids = np.unique(queries_np, return_inverse=True)
        label_ids = torch.from_numpy(label_ids).to(mask_pts.device)  # 确保在同一设备
        n_unique = len(unique_labels)  # 2. 合并mask_pts（兼容旧版PyTorch的实现）
        merged_mask_pts = torch.zeros(n_unique, mask_pts.size(1), 
                                    dtype=mask_pts.dtype, 
                                    device=mask_pts.device)
        for i in range(n_unique):  # 对每个唯一标签执行聚合操作
            mask = (label_ids == i)  # 找到属于当前标签的所有行索引
            if mask.any():
                merged_mask_pts[i] = mask_pts[mask].max(dim=0).values  # 取这些行的最大值（实现OR操作）
        merged_queries = unique_labels.tolist()  # 3. 整理结果
        labels = merged_queries  # 替换原本的标签
        mask_pts = merged_mask_pts
        
        point_dict = prep_points_train(pts_xyz.numpy(), pts_rgb.numpy(), normal.numpy(), mask_pts.numpy())

        ## encode label
        inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs) # n_masks, feat_dim (768)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        point_dict['label_embeds'] = text_feat

        
        return point_dict'''

class EvalData(Dataset):
    def __init__(self, data_root, split):
        assert split in ["val", "test", "train"]
        self.data_root = data_root
        self.obj_path_list = []
        with open(f"{data_root}/{split}.txt", "r") as f:
            self.obj_path_list = f.read().splitlines()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224").cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    
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
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    
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
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

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
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

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
