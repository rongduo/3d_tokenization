"""
如果从带decoder的中间特征进行测试，效果会怎么样呢
"""

import os
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
from model.evaluation.core import save_feats_to_ply, compute_3d_iou_upsample
from common.utils import visualize_pt_labels, visualize_pt_heatmap
import time
from model.evaluation.utils import set_seed, load_model
from transformers import AutoTokenizer, AutoModel

from pipes_eval.d3compat.data import Eval3dcom, collate_fn
from pipes.ours_token3d_acc.train_mutigpus import (
    attach_utonia_feature,
    init_utonia_runtime,
    str2bool,
)
from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder


def extract_three_outputs(model_outputs):
    if not isinstance(model_outputs, (tuple, list)):
        raise TypeError(
            f"模型输出格式需为元组或列表，当前为 {type(model_outputs)} "
            "请先确保model_outputs是可迭代的多输出格式（如model返回tuple/list）"
        )

    backbone_feat, decoder_out, decoder_offset = model_outputs[:3]
    output_len = len(model_outputs)
    if output_len >= 3:
        pass
    else:
        missing_vars = ["decoder_out", "decoder_offset"][:3 - output_len]
        print(
            f"模型输出仅{output_len}个结果，已提取所有结果，缺失的{missing_vars}用None填充"
        )

    if decoder_offset is None:
        decoder_offset = []
        print("提示：decoder_offset为None，已自动设为空前向（len=0）")

    return backbone_feat, decoder_out, decoder_offset


def computePre_3d_iou(
    model_outputs,
    data,
    cat,
    xyz_sub,
    xyz_full,
    gt_full,
    N_CHUNKS=1,
    visualize_seg=False,
    visualize_all_heatmap=False,
    xyz_visualization=None,
    savepath=None,
):
    backbone_feat, decoder_out, decoder_offset = extract_three_outputs(model_outputs)
    num_labels = len(decoder_offset)
    n_parts = data["label_embeds"].shape[0]

    logits_list = []
    start_idx = 0
    for i in range(num_labels):
        end_idx = decoder_offset[i].item()
        label_logits = decoder_out[start_idx:end_idx].view(-1, 1)
        logits_list.append(label_logits)
        start_idx = end_idx

    logits = torch.cat(logits_list, dim=1)
    threshold = 0.5
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits)

    pred_labels = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
    for i in range(probs.shape[0]):
        positive_labels = torch.where(probs[i] > threshold)[0]
        if len(positive_labels) == 0:
            pred_labels[i] = 0
        elif len(positive_labels) == 1:
            pred_labels[i] = positive_labels[0] + 1
        else:
            max_prob = -float("inf")
            selected_label = 0
            for label in positive_labels:
                prob = probs[i, label].item()
                if prob > max_prob:
                    max_prob = prob
                    selected_label = label
            pred_labels[i] = selected_label + 1

    xyz_full = xyz_full.squeeze()
    chunk_len = xyz_full.shape[0] // N_CHUNKS + 1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len * i : chunk_len * (i + 1)]
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1)) ** 2
        cur_dist = (dist_all.sum(dim=-1)) ** 0.5
        min_idxs = torch.min(cur_dist, 1)[1]
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list, axis=0)

    pred_full = pred_labels[all_nn_idxs].cpu()
    acc = ((pred_full == gt_full) * 1).sum() / pred_full.shape[0]
    pred_np = pred_full.numpy()
    label_np = gt_full.squeeze().numpy()

    if savepath is not None:
        assert savepath.endswith(".ply")
        pngsavepath = savepath.replace(".ply", ".png")
        caption_list = visualize_pt_labels(
            xyz_visualization.cpu(),
            pred_full.cpu(),
            save_path=savepath,
            save_rendered_path=pngsavepath,
        )
        savepath_gt = savepath.replace(".ply", "_gt.ply")
        pngsavepath_gt = pngsavepath.replace(".png", "_gt.png")
        caption_list = visualize_pt_labels(
            xyz_visualization.cpu(),
            gt_full.squeeze().cpu(),
            save_path=savepath_gt,
            save_rendered_path=pngsavepath_gt,
        )
        print("caption_list:", caption_list)
        featssavepath = savepath.replace("_gt.ply", "_feats.ply")
        save_feats_to_ply(backbone_feat, xyz_sub, featssavepath)

    part_ious = []
    for part in range(n_parts):
        I = np.sum(np.logical_and(pred_np == part + 1, label_np == part + 1))
        U = np.sum(np.logical_or(pred_np == part + 1, label_np == part + 1))
        if U > 0:
            part_ious.append(I / float(U))

    full_miou = np.mean(part_ious) if part_ious else 0.0
    full_macc = acc.item()
    return full_miou, full_macc


def evaluate3d(
    model,
    dataloader,
    panoptic=False,
    N_CHUNKS=1,
    visualize_seg=False,
    visualize_all_heatmap=False,
    savedir=None,
    testname=None,
    utonia_model=None,
    utonia_device=None,
):
    temperature = np.exp(model.ln_logit_scale.item())
    iou_full_list = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and "full" not in key:
                    data[key] = data[key].cuda(non_blocking=True)

            data["mask_offset"] = torch.tensor(
                [data["label_embeds"].shape[0]],
                device=data["offset"].device,
            )

            if utonia_model is not None:
                attach_utonia_feature(data, utonia_model, utonia_device)

            model_output = model(data)
            if isinstance(model_output, (tuple, list)):
                if len(model_output) == 0:
                    raise ValueError("模型输出为空元组/列表，无法提取net_out")
                net_out = model_output[0]
            elif isinstance(model_output, torch.Tensor):
                net_out = model_output
            else:
                raise TypeError(
                    f"不支持的模型输出格式：{type(model_output)}，"
                    "请扩展get_first_net_out函数以支持该格式（当前支持：张量、元组、列表、字典）"
                )

            text_embeds = data["label_embeds"]
            gt_full = data["gt_full"]
            xyz_sub = data["coord"]
            xyz_full = data["xyz_full"]
            cat = data["class_name"][0]
            savepath = None

            if testname == "feats":
                full_miou, _ = compute_3d_iou_upsample(
                    net_out,
                    text_embeds,
                    temperature,
                    cat,
                    xyz_sub,
                    xyz_full,
                    gt_full,
                    panoptic=panoptic,
                    N_CHUNKS=N_CHUNKS,
                    visualize_seg=visualize_seg,
                    savepath=savepath,
                    visualize_all_heatmap=visualize_all_heatmap,
                    xyz_visualization=data["xyz_visualization"],
                )
            else:
                full_miou, _ = computePre_3d_iou(
                    model_output,
                    data,
                    cat,
                    xyz_sub,
                    xyz_full=xyz_full,
                    gt_full=gt_full,
                    N_CHUNKS=N_CHUNKS,
                    visualize_seg=visualize_seg,
                    savepath=savepath,
                    visualize_all_heatmap=visualize_all_heatmap,
                    xyz_visualization=data["xyz_visualization"],
                )

            iou_full_list += [full_miou]

    full_miou = np.mean(iou_full_list)
    return full_miou


def eval_category_partnete(
    data_root,
    category,
    model,
    apply_rotation=False,
    subset=False,
    decorated=True,
    visualize_seg=False,
    visualize_all_heatmap=False,
    savedir=None,
    textembeds=None,
    datatype=None,
    testname=None,
    parts_suffix="",
    utonia_model=None,
    utonia_device=None,
):
    test_data = Eval3dcom(
        data_root,
        category,
        textembeds=textembeds,
        datatype=datatype,
        apply_rotation=apply_rotation,
        decorated=decorated,
        parts_suffix=parts_suffix,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
    stime = time.time()
    full_miou = evaluate3d(
        model,
        test_loader,
        panoptic=False,
        N_CHUNKS=20,
        visualize_seg=visualize_seg,
        visualize_all_heatmap=visualize_all_heatmap,
        savedir=savedir,
        testname=testname,
        utonia_model=utonia_model,
        utonia_device=utonia_device,
    )
    print(f"{category}: miou: {full_miou}")
    etime = time.time()
    print("time:", etime - stime, " s")
    return category, full_miou, etime - stime


def eval_partnete(
    data_root,
    model,
    apply_rotation,
    subset,
    decorated,
    textembeds=None,
    datatype=None,
    save_path=None,
    testname=None,
    parts_suffix="",
    utonia_model=None,
    utonia_device=None,
):
    categorynames = os.listdir(data_root)
    print("categorynames:", len(categorynames))

    results = []
    full_mious = []
    time_all = 0.0
    save_dir = os.path.dirname(save_path) if save_path is not None else None

    for cat in categorynames:
        print("--------------------------------------")
        print("cat:", cat)
        category, full_miou, time_cur = eval_category_partnete(
            data_root,
            cat,
            model,
            apply_rotation=apply_rotation,
            subset=subset,
            decorated=decorated,
            savedir=save_dir,
            textembeds=textembeds,
            datatype=datatype,
            testname=testname,
            parts_suffix=parts_suffix,
            utonia_model=utonia_model,
            utonia_device=utonia_device,
        )
        full_miou_rounded = round(full_miou, 4)
        results.append((category, full_miou_rounded))
        full_mious.append(full_miou)
        time_all += time_cur

    full_miou_avg = round(np.mean(full_mious), 4)
    print(f"{'类别':<15} {'iou':<10}")
    print("-------------------------")
    for category, miou in results:
        print(f"{category:<15} {miou:<10.4f}")
    print("-------------------------")
    print(f"{'平均值':<15} {full_miou_avg:<10.4f}")

    if save_path is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"{'类别':<15} {'iou':<10}\n")
            f.write("-------------------------\n")
            for category, miou in results:
                f.write(f"{category:<15} {miou:<10.4f}\n")
            f.write("-------------------------\n")
            f.write(f"{'平均值':<15} {full_miou_avg:<10.4f}\n")


if __name__ == "__main__":
    set_seed(123)
    parser = argparse.ArgumentParser(
        description="Please specify a benchmark name and evaluation configurations"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=str,
        help="The benchmark to evaluate on. Should be Objaverse, ShapeNetPart, or PartNetE",
    )
    parser.add_argument(
        "--data_root", required=True, type=str, help="Root directory of the benchmark data"
    )
    parser.add_argument(
        "--save_dir", required=True, type=str, help="results save to"
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="path of the checkpoint to evaluate",
    )
    parser.add_argument(
        "--d3com_datatype", required=True, type=str, help="coarse or fine"
    )
    parser.add_argument(
        "--net_type", required=True, type=str, help="使用的网络架构"
    )
    parser.add_argument(
        "--test_type", required=True, type=str, help="测试方式，feats or pre"
    )
    parser.add_argument(
        "--objaverse_split",
        type=str,
        help='If benchmark is Objaverse, specify "seenclass", "unseen" or "shapenetpart',
    )
    parser.add_argument(
        "--canonical",
        action="store_false",
        dest="rotate",
        help="whether to perform random rotation - this only applies to ShapeNetPart or PartNetE which have canonical orientations",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        dest="subsample",
        help="whether to evaluate on subset - this only applies to ShapeNetPart of PartNetE",
    )
    parser.add_argument(
        "--part_query",
        action="store_false",
        dest="decorate",
        help="if true, evaluate with {part} of a {object} as query prompt; if false, evaluate with {part} as query prompt",
    )
    parser.add_argument(
        "--use_shapenetpart_topk_prompt",
        action="store_true",
        help="This only applies to ShapeNetPart or Objaverse-ShapeNetPart. Whether to use the topk prompt following PointCLIPV2's procedures to choose prompts",
    )
    parser.add_argument(
        "--textembeds",
        type=str,
        default="clip",
        help="The type of text embeddings. (默认: clip)",
    )
    parser.add_argument(
        "--parts_suffix",
        default="",
        type=str,
        help="Suffix for part token files (e.g. '_partsam' -> parts_partsam.pt)",
    )
    parser.add_argument(
        "--enable_online_utonia",
        type=str2bool,
        default=False,
        help="Compute utonia_feat online (required for net8/partsam models)",
    )
    parser.set_defaults(
        rotate=True,
        subsample=False,
        decorate=True,
        use_shapenetpart_topk_prompt=False,
    )
    args = parser.parse_args()

    if args.net_type == "net1" or args.net_type == "net2":
        model = load_model(args.checkpoint_path)
    elif args.net_type == "net3":
        from release_pipeline3.stage1_semanspace.halfd3com_worot_aligncates_decoder.mixdecodernet import (
            PointSemSegWithDecoder,
        )

        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == "net4":
        from release_pipeline5ab.ab4_partfieldloss_sizeaug_decoder_canoncolor.mixdecodernet import (
            PointSemSegWithDecoder_test as PointSemSegWithDecoder,
        )

        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == "net5":
        from release_pipeline5ab.ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign.mixdecodernet import (
            PointSemSegWithDecoder_test as PointSemSegWithDecoder,
        )

        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == "net6":
        from release_pipeline5ab.ab6_partfieldloss_sizeaug_decoder_bbox.mixdecodernet import (
            PointSemSegWithbboxDecoder,
        )

        model = PointSemSegWithbboxDecoder(args=args)
    elif args.net_type == "net7":
        from release_pipeline6.ab1_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox.mixdecodernet import (
            PointSemSegWithDecoder,
        )

        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == "net8":
        from release_module.network.tokenbased_pre import PointSemSegWithDecoder

        model = PointSemSegWithDecoder(args=args)

    model.load_state_dict(
        torch.load(args.checkpoint_path)["model_state_dict"], strict=True
    )
    model.eval()
    model = model.cuda()

    utonia_model, utonia_device = init_utonia_runtime(0, args.enable_online_utonia)
    if args.enable_online_utonia:
        print(f"Online Utonia enabled on {utonia_device}")
    if args.parts_suffix:
        print(f"Using part token files: parts{args.parts_suffix}.pt")

    if args.benchmark == "d3compat":
        dataset = "d3compat"
        netname = args.net_type
        testname = args.test_type
        d3comname = args.d3com_datatype
        traindataname = args.checkpoint_path.split("/")[1]
        save_pat = os.path.join(
            args.save_dir,
            f"{dataset}_{traindataname}_{netname}_{d3comname}_{testname}",
        )
        if args.decorate:
            save_pat = save_pat + "_partcates_"
        if args.rotate:
            save_pat = save_pat + "_roted_"
        save_path = save_pat + ".txt"
        eval_partnete(
            args.data_root,
            model,
            apply_rotation=args.rotate,
            subset=args.subsample,
            decorated=args.decorate,
            textembeds=args.textembeds,
            datatype=args.d3com_datatype,
            save_path=save_path,
            testname=testname,
            parts_suffix=args.parts_suffix,
            utonia_model=utonia_model,
            utonia_device=utonia_device,
        )
    else:
        print("Invalid benchmark. Please choose d3compat for this script.")
