# from torch.amp import GradScaler, autocast
# scaler = GradScaler('cuda')

########### 最后30轮去掉canonical color loss

import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn  # 补充导入nn模块（原代码缺失）
from model.backbone.pt3.model import PointSemSeg
from tqdm import tqdm
import numpy as np
from model.evaluation.core import viz_pred_mask, compute_overall_iou_objwise
from model.training.loss import DistillLossContrastive, FieldDistillLossContrastive
from transformers import AutoTokenizer, AutoModel
import random
from model.evaluation.core import visualize_3d_upsample

from pipes.ours_token3d_acc.data import TrainingData, EvalData, collate_fn
from pipes.ours_token3d_acc.mixdecoderloss import BalancedMaskCrossEntropyLoss,CanonicalColorLoss
from pipes.ours_token3d_acc.mixdecoderloss_bboxoff import PointBasedBBoxOffsetLoss


from release_module.network.tokenbased_pre import PointSemSegWithDecoder

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def create_data_loader(rank, data_root, shuffle_train, shuffle_test,
                       eval_split, drop_last_train=True, drop_last_test=False,
                       is_test_only=False, batch_size=16,
                       train_num_workers=2, eval_num_workers=0,
                       pin_memory=True, persistent_workers=True,
                       prefetch_factor=2):
    test_data = EvalData(data_root, split=eval_split)
    test_sampler = DistributedSampler(
        test_data,
        shuffle=shuffle_test
    ) if dist.is_initialized() else None

    test_loader_kwargs = dict(
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and eval_num_workers > 0,
    )
    if eval_num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = prefetch_factor

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=shuffle_test and (test_sampler is None),
        collate_fn=collate_fn,
        drop_last=drop_last_test,
        sampler=test_sampler,
        **test_loader_kwargs
    )

    if is_test_only:
        return test_loader, test_sampler

    train_data = TrainingData(data_root)
    BS = min(batch_size, len(train_data))
    if rank == 0:
        print(f'Batch size per GPU: {BS}')

    train_sampler = DistributedSampler(
        train_data,
        shuffle=shuffle_train
    ) if dist.is_initialized() else None

    train_loader_kwargs = dict(
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and train_num_workers > 0,
    )
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_data,
        batch_size=BS,
        shuffle=shuffle_train and (train_sampler is None),
        collate_fn=collate_fn,
        drop_last=drop_last_train,
        sampler=train_sampler,
        **train_loader_kwargs
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    total_batch_size = BS * world_size
    train_iter_per_epoch = (len(train_data) // total_batch_size) + 1

    return train_loader, test_loader, train_iter_per_epoch, train_sampler


def evaluate(rank, model, dataloader, loss_fn, n_epoch, set_name,
             eval_loss=True, visualize_idxs=[20, 25, 55, 80, 139]):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    n_visualize_epoch = 5
    prefix = "pt"
    iou_list = []
    loss_list = []
    i = 0

    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f"Evaluating {set_name}-set (GPU {rank})") if rank == 0 else dataloader

        for data in iterator:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(rank, non_blocking=True)

            # 先获取模型输出，再单独获取ln_logit_scale，兼容不支持return_logit_scale的模型
            net_out = model(x=data)
            # 根据是否为DDP模型获取ln_logit_scale
            if world_size > 1:
                ln_logit_scale = model.module.ln_logit_scale
            else:
                ln_logit_scale = model.ln_logit_scale
            temperature = torch.exp(ln_logit_scale)

            text_embeds = data['label_embeds']
            masks = data['masks']
            mask_view_idxs = data["mask_view_idxs"]
            point2face = data['point2face']
            pix2face = data['pixel2face']
            labels = data['labels']
            mask_pts = data['mask2pt']
            pt_offset = data['offset']

            m = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(rank)
            tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
            inputs = tokenizer(labels[0], padding="max_length", return_tensors="pt").to(rank)

            with torch.no_grad():
                text_feat = m.get_text_features(** inputs)

            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

            iou = compute_overall_iou_objwise(
                pred=net_out,
                text_embeds=text_embeds,
                masks=masks,
                mask_view_idxs=mask_view_idxs,
                point2face=point2face,
                pixel2face=pix2face,
                temperature=temperature
            )
            iou_list.append(iou)

            if eval_loss:
                loss = loss_fn(
                    net_out,
                    pt_offset,
                    text_embeds,
                    mask_pts,
                    ln_logit_scale
                )
                loss_list.append(loss.item())

            if rank == 0 and n_epoch % n_visualize_epoch == 0 and (i in visualize_idxs):
                viz_pred_mask(
                    pred=net_out,
                    text_embeds=text_embeds,
                    texts=[[x] for x in labels[0]],
                    masks=masks,
                    mask_view_idxs=mask_view_idxs,
                    point2face=point2face,
                    pixel2face=pix2face,
                    n_epoch=n_epoch,
                    obj_visualize_idx=i,
                    prefix=f"{prefix}-{set_name}",
                    temperature=temperature
                )
            i += 1

    if world_size > 1:
        iou_tensor = torch.tensor(iou_list, device=rank)
        all_iou = [torch.zeros_like(iou_tensor) for _ in range(world_size)]
        dist.all_gather(all_iou, iou_tensor)
        iou_list = [item.item() for sublist in all_iou for item in sublist]

        if eval_loss:
            loss_tensor = torch.tensor(loss_list, device=rank)
            all_loss = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            dist.all_gather(all_loss, loss_tensor)
            loss_list = [item.item() for sublist in all_loss for item in sublist]

    miou = np.mean(iou_list) if iou_list else 0.0
    loss = np.mean(loss_list) if loss_list and eval_loss else 0.0
    return miou, loss


def save_ply(points, save_path):
    """
    保存点云为 PLY 格式（仅包含坐标）
    points: 形状为 [N, 3] 的 numpy 数组，每行对应 (x, y, z)
    save_path: 保存路径（如 'output.ply'）
    """
    n_points = points.shape[0]

    # 写入 PLY 头部信息
    with open(save_path, 'w') as f:
        # PLY 格式标识
        f.write("ply\n")
        f.write("format ascii 1.0\n")  # 使用 ASCII 格式（可读性强，二进制可选）
        f.write(f"element vertex {n_points}\n")  # 顶点数量
        # 定义坐标属性（x, y, z）
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")  # 头部结束

        # 写入点坐标
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def set_baseline_freeze_status(model, freeze: bool, world_size: int):
    """
    设置baseline（backbone）的冻结状态
    Args:
        model: 完整模型（可能是DDP包装的）
        freeze: True=冻结，False=解冻
        world_size: 进程数（用于判断是否为DDP模型）
    """
    # 获取原始模型（DDP包装时需通过model.module访问）
    raw_model = model.module if world_size > 1 else model
    
    # 冻结/解冻backbone的所有参数
    for param in raw_model.backbone.parameters():
        param.requires_grad = not freeze  # freeze=True → requires_grad=False
    
    # 可选：冻结/解冻ln_logit_scale（如果属于baseline的一部分）
    if hasattr(raw_model, 'ln_logit_scale'):
        raw_model.ln_logit_scale.requires_grad = not freeze


def set_backbone_only_train_status(model, backbone_only: bool, world_size: int):
    """
    设置是否仅训练backbone
    Args:
        model: 完整模型（可能是DDP包装的）
        backbone_only: True=仅训练backbone与ln_logit_scale，False=训练全部参数
        world_size: 进程数（用于判断是否为DDP模型）
    """
    raw_model = model.module if world_size > 1 else model

    for name, param in raw_model.named_parameters():
        if "backbone." in name or "ln_logit_scale" in name:
            param.requires_grad = True
        else:
            param.requires_grad = not backbone_only


# ==============================================================================
# 新增：梯度回传检查函数（核心功能）
# ==============================================================================
def check_unused_parameters(raw_model, current_epoch, current_iter, rank=0):
    """
    检查模型中未参与梯度回传的参数，并按模块分类打印
    关键判断：requires_grad=True（预期训练）但 grad=None（未参与回传）的参数
    Args:
        raw_model: 原始模型（非DDP包装）
        current_epoch: 当前epoch（用于日志定位）
        current_iter: 当前迭代（用于日志定位）
        rank: GPU编号（仅rank=0打印，避免多GPU重复输出）
    """
    if rank != 0:
        return

    # 按网络模块分类参数（根据参数名称前缀匹配）
    module_params = {
        "Backbone（骨干网络）": [],
        "Decoder（解码器）": [],
        "CanonColor Head（颜色损失头）": [],
        "BBox Head（边界框损失头）": [],
        "Other（其他参数，如ln_logit_scale）": []
    }

    # 遍历所有参数，按模块分类
    for name, param in raw_model.named_parameters():
        if "backbone." in name:
            module_params["Backbone（骨干网络）"].append((name, param))
        elif "decoder." in name:
            module_params["Decoder（解码器）"].append((name, param))
        elif "canoncolor" in name or "canoncial_color" in name:  # 匹配颜色相关头
            module_params["CanonColor Head（颜色损失头）"].append((name, param))
        elif "bbox" in name:  # 匹配边界框相关头
            module_params["BBox Head（边界框损失头）"].append((name, param))
        else:
            module_params["Other（其他参数，如ln_logit_scale）"].append((name, param))

    # 统计未参与回传的参数
    unused_params = {}
    frozen_params = {}  # 预期冻结的参数（requires_grad=False）
    for module_name, params in module_params.items():
        unused = []
        frozen = []
        for name, param in params:
            if param.requires_grad:
                # 预期训练但无梯度 → 未参与回传
                if param.grad is None:
                    unused.append(name)
            else:
                # 预期冻结（requires_grad=False）→ 正常情况
                frozen.append(name)
        if unused:
            unused_params[module_name] = unused
        if frozen:
            frozen_params[module_name] = frozen

    # 格式化打印日志
    print("="*80)
    print(f"【梯度回传检查】Epoch: {current_epoch:3d} | Iter: {current_iter:4d} | GPU: {rank}")
    print("="*80)

    # 1. 打印预期冻结的参数（参考信息，非问题）
    print("\n1. 预期冻结的参数（requires_grad=False，正常情况）：")
    has_frozen = False
    for module_name, params in frozen_params.items():
        if params:
            has_frozen = True
            print(f"   {module_name}:")
            for param_name in params[:5]:  # 只打印前5个，避免日志过长
                print(f"     - {param_name}")
            if len(params) > 5:
                print(f"     - ... 共{len(params)}个参数")
    if not has_frozen:
        print("   无预期冻结的参数")

    # 2. 打印未参与回传的参数（问题点，需关注）
    print("\n2. 未参与梯度回传的参数（requires_grad=True但grad=None，需检查）：")
    has_unused = False
    for module_name, params in unused_params.items():
        if params:
            has_unused = True
            print(f"   ⚠️ {module_name}:")
            for param_name in params[:10]:  # 只打印前10个，避免日志过长
                print(f"     - {param_name}")
            if len(params) > 10:
                print(f"     - ... 共{len(params)}个参数")
    if not has_unused:
        print("   ✅ 所有预期训练的参数均参与梯度回传")

    print("="*80 + "\n")


def train(rank, world_size, args):
    setup(rank, world_size)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    if rank == 0:
        ckpt_dir = os.path.join(args.ckpt_dir, f"find3d_{args.exp_suffix}")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        ckpt_dir = None

    # 初始化包含decoder的完整模型
    model = PointSemSegWithDecoder(args=args)
    model = model.to(rank)

    # **1. 加载预训练权重（仅backbone）**
    if args.pretrained_path:
        if rank == 0:
            
            print(f"Loading pretrained backbone from {args.pretrained_path}")
            pretrained_weights = torch.load(args.pretrained_path)
            # 提取backbone相关权重（过滤掉decoder参数）
            backbone_weights = {
                k.replace('backbone.', ''): v  # 适配新模型的参数命名
                for k, v in pretrained_weights["model_state_dict"].items()
                if k.startswith('backbone.') or not 'decoder.' in k
            }
            # 加载到backbone（忽略decoder参数不匹配的错误）
            model.backbone.load_state_dict(backbone_weights, strict=False)
            print("Pretrained backbone loaded")

            '''#### debug 这里能不能正常加载全部的权重呢 ; 这个是可以的
            pretrained_weights = torch.load(args.pretrained_path)
            print('pretrained_weights:',pretrained_weights.keys())
            print('pretrained_weights:',pretrained_weights["optimizer_state_dict"])
            print('pretrained_weights:',pretrained_weights["total_loss"])
            model.load_state_dict(pretrained_weights["model_state_dict"])
            print('succeed loaded pretrain model')
            asdf'''



        # 关键：将rank 0的参数广播到所有进程（包含backbone和decoder的初始参数）
        if world_size > 1:
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            if rank == 0:
                print("Pretrained parameters broadcast to all ranks")

    # **2. 初始训练状态**
    freeze_baseline = args.freeze_baseline_epochs > 0
    backbone_only_mode = args.backbone_only_epochs > 0
    if backbone_only_mode:
        set_backbone_only_train_status(model, backbone_only=True, world_size=1)
        freeze_baseline = False
        if rank == 0:
            print(f"Initial state: Backbone-ONLY training for first {args.backbone_only_epochs} epochs")
    else:
        set_baseline_freeze_status(model, freeze=freeze_baseline, world_size=1)  # 此时尚未包装DDP
        if rank == 0:
            print(f"Initial state: Baseline (backbone) is {'FROZEN' if freeze_baseline else 'UNFROZEN'} "
                  f"for first {args.freeze_baseline_epochs} epochs")

    # 最后包装DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            find_unused_parameters=args.find_unused_parameters,
        )

    # **3. 数据加载**
    train_loader, test_loader, train_iter_per_epoch, train_sampler = create_data_loader(
        rank, args.data_root,
        shuffle_train=True, shuffle_test=False,
        eval_split="val", drop_last_train=True,
        drop_last_test=False, batch_size=args.batch_size,
        train_num_workers=args.train_num_workers,
        eval_num_workers=args.eval_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )

    train_val_loader, _ = create_data_loader(
        rank, args.data_root,
        shuffle_train=True, shuffle_test=False,
        eval_split="train", is_test_only=True,
        eval_num_workers=args.eval_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )

    # **4. 优化器初始化**
    # 注意：即使backbone参数被冻结，优化器仍需接收所有参数（冻结参数的requires_grad=False会被自动跳过）
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # **5. 从checkpoint继续训练（处理冻结状态）**
    start_epoch = 1  # 初始epoch
    # 补充定义decoder_loss（原代码保存ckpt时引用但未初始化，避免报错）
    decoder_loss = torch.tensor(0.0, device=rank)
    if args.continue_path:
        if rank == 0:
            print(f"Continuing training from {args.continue_path}")
            checkpoint = torch.load(args.continue_path)
            # 加载完整模型参数（包含backbone和decoder）
            model.load_state_dict(checkpoint["model_state_dict"])
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1  # 从上次结束的下一个epoch开始
            # 修复ln_logit_scale参数
            raw_model = model.module if world_size > 1 else model
            raw_model.ln_logit_scale = nn.Parameter(checkpoint["lntemperature"].data, requires_grad=True)
            # 恢复decoder_loss（避免保存时报错）
            decoder_loss = checkpoint.get("decoder_loss", torch.tensor(0.0, device=rank))
            print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, resuming from epoch {start_epoch}")

        if world_size > 1:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        dist.broadcast(v, src=0)

        # 根据继续训练的起始epoch更新训练状态
        if args.backbone_only_epochs > 0:
            if start_epoch > args.backbone_only_epochs:
                backbone_only_mode = False
                set_backbone_only_train_status(model, backbone_only=False, world_size=world_size)
                if rank == 0:
                    print(f"Resumed from epoch {start_epoch} > backbone_only_epochs {args.backbone_only_epochs}, "
                          f"switch to full-parameter training")
            else:
                backbone_only_mode = True
                set_backbone_only_train_status(model, backbone_only=True, world_size=world_size)
                if rank == 0:
                    print(f"Resumed in backbone-only phase (<= {args.backbone_only_epochs} epochs)")
        elif start_epoch > args.freeze_baseline_epochs:
            freeze_baseline = False
            set_baseline_freeze_status(model, freeze=freeze_baseline, world_size=world_size)
            if rank == 0:
                print(f"Resumed from epoch {start_epoch} > freeze_baseline_epochs {args.freeze_baseline_epochs}, "
                      f"Baseline (backbone) is UNFROZEN")

    # **6. 学习率调度器**
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        train_iter_per_epoch * args.n_epoch,
        eta_min=args.eta_min
    )

    # **7. 损失函数初始化**
    # criterion = DistillLossContrastive()  # 原对比损失（baseline loss）
    criterion = FieldDistillLossContrastive()  # 更新对比损失
    # label_criterion = BalancedMaskCrossEntropyLoss().cuda(rank)  # 解码器损失
    # decoder_loss_weight = 10.0
    canoncolor_criterion = CanonicalColorLoss().cuda(rank)   # 规范空间的颜色损失
    canoncolor_loss_weight = 10.0  # decoder loss权重设为10
    bbox_criterion = PointBasedBBoxOffsetLoss().cuda(rank)  #  part 包围盒损失
    # bbox_loss_weight = 1.3
    bbox_loss_weight = 3

    # **8. 训练主循环**
    global_iter = 0
    model.train()
    # 标记是否已执行过解冻操作（避免重复解冻）
    has_unfrozen = not freeze_baseline
    has_switched_from_backbone_only = not backbone_only_mode

    for epoch in range(start_epoch, args.n_epoch + 1):
        current_epoch = epoch

        # **关键：epoch开始时检查是否需要切换训练阶段**
        if args.backbone_only_epochs > 0:
            if not has_switched_from_backbone_only and current_epoch > args.backbone_only_epochs:
                backbone_only_mode = False
                set_backbone_only_train_status(model, backbone_only=False, world_size=world_size)
                has_switched_from_backbone_only = True
                if rank == 0:
                    print(f"=== Epoch {current_epoch}: Switch to full-parameter training ===")
        else:
            if not has_unfrozen and current_epoch > args.freeze_baseline_epochs:
                freeze_baseline = False
                set_baseline_freeze_status(model, freeze=freeze_baseline, world_size=world_size)
                has_unfrozen = True
                if rank == 0:
                    print(f"=== Epoch {current_epoch}: Unfreeze Baseline (backbone) ===")

        # 打印当前epoch的冻结状态（仅rank 0）
        if rank == 0:
            if args.backbone_only_epochs > 0:
                phase = "BACKBONE-ONLY" if backbone_only_mode else "FULL-PARAM"
                print(f"\n=== Training Epoch {current_epoch}/{args.n_epoch} | Training Phase: {phase} ===")
            else:
                print(f"\n=== Training Epoch {current_epoch}/{args.n_epoch} | "
                      f"Baseline Status: {'FROZEN' if freeze_baseline else 'UNFROZEN'} ===")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_epoch_current = []

        train_iterator = tqdm(
            train_loader,
            desc=f"Training epoch: {current_epoch}/{args.n_epoch} (GPU {rank})"
        ) if rank == 0 else train_loader

        for iter_idx, data in enumerate(train_iterator):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(rank, non_blocking=True)

            mask_points = data['mask2pt']
            mask_embeds = data['label_embeds']
            pt_offset = data['offset']

            # with autocast('cuda'): 
            if 1:
                # 获取模型输出（backbone特征 + decoder输出 + decoder偏移量）
                backbone_feat, decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset = model(data)

                # 根据是否为DDP模型获取原始模型（用于参数检查）
                raw_model = model.module if world_size > 1 else model
                ln_logit_scale = raw_model.ln_logit_scale

                # 计算baseline loss（原对比损失）
                baseline_loss = criterion(
                    backbone_feat,
                    pt_offset,
                    mask_embeds,
                    mask_points,
                    ln_logit_scale
                )

                # 计算规范空间的颜色损失
                gt_color = data['canoncial_color']
                canoncolor_loss = canoncolor_criterion(canoncolor_out, gt_color, pt_offset, mask_points)
                weighted_canoncolor_loss = canoncolor_loss * canoncolor_loss_weight

                # 计算bbox loss
                bbox_loss = bbox_criterion(
                    bbox_pred=bbox_pred,
                    pts=data['coord'],
                    pt_offset=pt_offset,
                    mask_points=mask_points
                )
                weighted_bbox_loss = bbox_loss_weight * bbox_loss

                # 计算总损失 - 修改部分：在最后30轮训练时去掉canoncolor_loss
                if current_epoch > args.n_epoch - 30:
                    total_loss = baseline_loss + weighted_bbox_loss
                    if rank == 0 and iter_idx == 0:
                        print(f"=== Epoch {current_epoch}: Removing canoncolor_loss (last 30 epochs) ===")
                else:
                    total_loss = baseline_loss + weighted_canoncolor_loss + weighted_bbox_loss
                
            loss_epoch_current.append(total_loss.item())
            cur_lr = scheduler.get_last_lr()[0]

            # 反向传播（冻结的backbone参数不会更新）
            opt.zero_grad()
            try:
                total_loss.backward()  # 计算梯度
                
                # =========================================================================
                # 关键修改：反向传播后检查未参与回传的参数（每个epoch的第1个迭代打印，避免日志冗余）
                # =========================================================================
                if args.check_grad_flow and iter_idx == 0:
                    check_unused_parameters(
                        raw_model=raw_model,
                        current_epoch=current_epoch,
                        current_iter=global_iter,
                        rank=rank
                    )
                
                opt.step()  # 更新参数
            except RuntimeError as e:
                if "element 0 of tensors does not require grad" in str(e):
                    # 定义日志文件路径
                    log_dir = "/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results/logs"
                    log_file = os.path.join(log_dir, "backuperror.txt")

                    # 确保日志目录存在（若不存在则创建）
                    os.makedirs(log_dir, exist_ok=True)

                    # 构建错误信息内容
                    error_msg = [
                        f"\n===== 错误时间: =====",
                        f"Epoch: {current_epoch}, Iter: {iter_idx}, GPU: {rank}",
                        f"错误详情: {str(e)}",
                        f"backbone_feat.requires_grad: {backbone_feat.requires_grad}",
                        f"total_loss.grad_fn: {total_loss.grad_fn}",
                        f"模型训练模式: {model.training}",
                        f"Baseline Status: {'FROZEN' if freeze_baseline else 'UNFROZEN'}",
                        "关键参数梯度状态:"
                    ]
                    # 添加参数信息
                    for name, param in raw_model.named_parameters():
                        if "ln_logit_scale" in name or "backbone" in name or "decoder" in name:
                            error_msg.append(f"  - {name}: requires_grad={param.requires_grad}")
                    error_msg.append("=========================================\n")

                    # 写入文件（追加模式，避免覆盖历史日志）
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\n".join(error_msg))

                    # 同时打印到控制台（便于实时观察）
                    print("\n".join(error_msg))
                    print("跳过当前迭代，继续训练...\n")
                    continue
                else:
                    raise e
            scheduler.step()

            # 打印迭代日志（包含损失分解）
            if rank == 0:
                temperature = np.exp(ln_logit_scale.item())
                # 修改日志打印：根据是否包含canoncolor_loss调整输出
                if current_epoch > args.n_epoch - 30:
                    print(f"iter {global_iter:4d} | "
                          f"baseline_loss: {baseline_loss.item():.4f} | "
                          f"bbox_loss(×{bbox_loss_weight}): {weighted_bbox_loss.item():.4f} (raw: {bbox_loss.item():.4f}) | "
                          f"total_loss: {total_loss.item():.4f} | "
                          f"lr: {cur_lr:.6f} | "
                          f"temp: {temperature:.2f} | "
                          f"STATUS: canoncolor_loss REMOVED (last 30 epochs)")
                else:
                    print(f"iter {global_iter:4d} | "
                          f"baseline_loss: {baseline_loss.item():.4f} | "
                          f"canoncolor_loss(×{canoncolor_loss_weight}): {weighted_canoncolor_loss.item():.4f} (raw: {canoncolor_loss.item():.4f}) | "
                          f"bbox_loss(×{bbox_loss_weight}): {weighted_bbox_loss.item():.4f} (raw: {bbox_loss.item():.4f}) | "
                          f"total_loss: {total_loss.item():.4f} | "
                          f"lr: {cur_lr:.6f} | "
                          f"temp: {temperature:.2f}")

            global_iter += 1

        # 计算epoch平均损失
        if world_size > 1:
            loss_tensor = torch.tensor(np.mean(loss_epoch_current), device=rank)
            all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            dist.all_gather(all_losses, loss_tensor)
            epoch_loss_avg = np.mean([loss.item() for loss in all_losses])
        else:
            epoch_loss_avg = np.around(np.mean(loss_epoch_current), decimals=4)

        # 保存完整模型参数（包含backbone和decoder）
        if rank == 0 and current_epoch % 5 == 0:
            ckpt_path_test = os.path.join(ckpt_dir, f"ckpt_{current_epoch}.pth")
            save_dict = {
                'epoch': current_epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'total_loss': total_loss,
                'baseline_loss': baseline_loss,
                'decoder_loss': decoder_loss,  # 补充decoder_loss，避免保存时KeyError
                'lntemperature': ln_logit_scale,
                'scheduler_state_dict': scheduler.state_dict(),
                'baseline_frozen': freeze_baseline  # 保存当前baseline冻结状态
            }
            torch.save(save_dict, ckpt_path_test)
            print(f"完整模型参数已保存至: {ckpt_path_test}")


        # 每轮训练结束后，彻底清理内存
        torch.cuda.empty_cache()
        # 手动删除可能残留的中间变量
        del backbone_feat, decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset, total_loss
        # 等待GPU完成当前计算，确保内存释放
        torch.cuda.synchronize()


    # 训练结束保存最终模型
    if rank == 0:
        final_ckpt_path = os.path.join(ckpt_dir, "ckpt_final.pth")
        final_save_dict = {
            'epoch': args.n_epoch,
            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'total_loss': total_loss,
            'baseline_loss': baseline_loss,
            'decoder_loss': decoder_loss,  # 补充decoder_loss
            'lntemperature': ln_logit_scale,
            'scheduler_state_dict': scheduler.state_dict(),
            'baseline_frozen': freeze_baseline
        }
        torch.save(final_save_dict, final_ckpt_path)
        print(f"最终完整模型参数已保存至: {final_ckpt_path}")

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    # 原有参数
    parser.add_argument('--n_epoch', type=int, required=True, help="Total training epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size per GPU")
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='Learning rate')
    parser.add_argument('--eta_min', type=float, default=0.00005, metavar='LR', help='Minimum learning rate')
    parser.add_argument('--step', type=int, default=40, help='LR decay step (unused for CosineAnnealing)')
    parser.add_argument('--use_aug', type=int, default=1, choices=[0, 1], help="Use data augmentation")
    parser.add_argument('--normalize_cloud', type=int, default=1, choices=[0, 1], help="Normalize point cloud")
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument('--continue_path', type=str, default=None, help="Path to resume training from checkpoint")
    parser.add_argument('--n_mov_avg', type=int, default=5, help="Number of moving average steps")
    parser.add_argument('--exp_suffix', default='', type=str, help="Experiment suffix for checkpoint directory")
    parser.add_argument('--data_root', required=True, type=str, help="Data root directory")
    parser.add_argument('--pretrained_path', type=str, default=None, help="Path to pretrained backbone model")
    
    # **新增超参数：控制baseline冻结轮数**
    parser.add_argument('--freeze_baseline_epochs', type=int, default=5, 
                        help="Number of epochs to freeze baseline (backbone) weights, default=5")
    parser.add_argument('--backbone_only_epochs', type=int, default=0,
                        help="Number of initial epochs to train only backbone (and ln_logit_scale), default=0")
    
    # =========================================================================
    # 新增：梯度检查开关（默认开启）
    # =========================================================================
    parser.add_argument('--check_grad_flow', type=str2bool, default=False, 
                        help="Whether to check unused parameters (no gradient flow), default=False")
    parser.add_argument('--find_unused_parameters', type=str2bool, default=False,
                        help="DDP find_unused_parameters flag, default=False for better performance")
    parser.add_argument('--train_num_workers', type=int, default=8,
                        help="DataLoader num_workers for training set")
    parser.add_argument('--eval_num_workers', type=int, default=2,
                        help="DataLoader num_workers for eval set")
    parser.add_argument('--pin_memory', type=str2bool, default=True,
                        help="Enable DataLoader pin_memory")
    parser.add_argument('--persistent_workers', type=str2bool, default=True,
                        help="Enable DataLoader persistent_workers")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help="DataLoader prefetch_factor when num_workers>0")

    args = parser.parse_args()
    args.seed = 123

    # 验证超参数合理性
    if args.freeze_baseline_epochs < 0 or args.freeze_baseline_epochs >= args.n_epoch:
        raise ValueError(f"--freeze_baseline_epochs ({args.freeze_baseline_epochs}) must be between 0 and --n_epoch ({args.n_epoch})-1")
    if args.backbone_only_epochs < 0 or args.backbone_only_epochs >= args.n_epoch:
        raise ValueError(f"--backbone_only_epochs ({args.backbone_only_epochs}) must be between 0 and --n_epoch ({args.n_epoch})-1")
    if args.freeze_baseline_epochs > 0 and args.backbone_only_epochs > 0:
        raise ValueError("--freeze_baseline_epochs and --backbone_only_epochs cannot both be > 0")

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} available GPU(s)")
    print(f"Configuration: freeze_baseline_epochs={args.freeze_baseline_epochs}, "
          f"backbone_only_epochs={args.backbone_only_epochs}, total epochs={args.n_epoch}")
    if args.train_num_workers < 0 or args.eval_num_workers < 0:
        raise ValueError("DataLoader num_workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1")
    if args.train_num_workers == 0 and args.persistent_workers:
        if world_size == 1:
            print("Warning: train_num_workers=0, forcing persistent_workers=False")
        args.persistent_workers = False

    print(f"Gradient flow check: {'Enabled' if args.check_grad_flow else 'Disabled'}")
    print(f"DDP find_unused_parameters: {args.find_unused_parameters}")
    print("DataLoader config: "
          f"train_workers={args.train_num_workers}, eval_workers={args.eval_num_workers}, "
          f"pin_memory={args.pin_memory}, persistent_workers={args.persistent_workers}, "
          f"prefetch_factor={args.prefetch_factor}")
    print(f"Note: canoncolor_loss will be removed in the last 30 epochs")

    if world_size > 1:
        print('----------using DDP---------------')
        mp.spawn(
            train,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        print('------------single gpu---------------')
        train(0, 1, args)


if __name__ == '__main__':
    main()
