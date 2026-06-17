



import os

import torch
import argparse
from torch.utils.data import DataLoader
from model.data.data import EvalData3D, EvalShapeNetPart, EvalPartNetE, collate_fn
import numpy as np
from pipes_eval.shapnetpart.core import compute_3d_iou_upsample, compute_3d_iou_upsample_pro
import time
# from pipes_eval.shapnetpart.utils import set_seed, load_model, muti_load_model
from model.evaluation.utils import set_seed, load_model

######## 对网络预测结果进行评价
def extract_three_outputs(model_outputs):
    """
    从模型输出中提取前3个结果，分别赋值给backbone_feat、decoder_out、decoder_offset
    若输出长度不足3个，缺失的变量用None填充（避免解包报错），并给出提示
    Args:
        model_outputs: 模型原始输出（支持元组、列表格式）
    Returns:
        backbone_feat: 第1个输出（通常为 backbone 特征）
        decoder_out: 第2个输出（通常为 decoder 主输出）
        decoder_offset: 第3个输出（通常为 offset 输出，用于后续计算num_labels）
    """
    # 1. 校验输入格式（仅支持元组/列表，避免非迭代类型报错）
    if not isinstance(model_outputs, (tuple, list)):
        raise TypeError(
            f"模型输出格式需为元组或列表，当前为 {type(model_outputs)} "
            "请先确保model_outputs是可迭代的多输出格式（如model返回tuple/list）"
        )
    
    # 2. 提取前3个输出，不足3个时用None填充（保证解包时长度为3）
    # 切片取前3个：model_outputs[:3] → 长度最多3；再用[:3]和+[None]*3确保总长度固定为3
    # three_outputs = ( + [None] * 3)[:3]
    backbone_feat, decoder_out, decoder_offset = model_outputs[:3]
    
    # 3. 输出提示（便于调试，确认当前输出长度和填充情况）
    output_len = len(model_outputs)
    if output_len >= 3:
        a=1
        # print(f"模型输出共{output_len}个结果，已提取前3个分别赋值给backbone_feat、decoder_out、decoder_offset")
    else:
        missing_vars = ["decoder_out", "decoder_offset"][:3 - output_len]  # 缺失的变量名
        print(f"模型输出仅{output_len}个结果，已提取所有结果，缺失的{missing_vars}用None填充")
    
    # 4. 处理decoder_offset为None的情况（避免后续len(decoder_offset)报错）
    if decoder_offset is None:
        # 若decoder_offset缺失，可根据业务逻辑设置默认值（如空列表/空张量）
        # 示例1：设为空前向（不影响后续len计算，len([])=0）
        decoder_offset = []
        # 示例2：若需匹配张量格式，可设为空张量（需与模型输出的设备/ dtype一致）
        # decoder_offset = torch.tensor([], device=backbone_feat.device if backbone_feat is not None else 'cpu')
        print("提示：decoder_offset为None，已自动设为空前向（len=0）")
    
    return backbone_feat, decoder_out, decoder_offset
def computePre_3d_iou(model_outputs,  # 改为接收模型输出: (backbone_feat, decoder_out, decoder_offset)
        data,           # 新增参数，用于获取label_embeds
        cat,
        xyz_sub,
        xyz_full,       # n_pts, 3
        gt_full,        # n_pts,
        N_CHUNKS=1,
        visualize_seg=False,
        visualize_all_heatmap=False,
        xyz_visualization=None,
        savepath=None
        ):
    # 解析模型输出
    backbone_feat, decoder_out, decoder_offset = extract_three_outputs(model_outputs)
    num_labels = len(decoder_offset)
    
    # 获取标签数量
    n_parts = data['label_embeds'].shape[0]
    
    # 解析每个标签的预测结果
    logits_list = []
    start_idx = 0
    for i in range(num_labels):
        end_idx = decoder_offset[i].item()
        # 提取当前标签的预测结果并调整形状
        label_logits = decoder_out[start_idx:end_idx].view(-1, 1)
        logits_list.append(label_logits)
        start_idx = end_idx
    
    # 将所有标签的预测结果拼接起来，形成[n_pts, n_labels]的张量
    logits = torch.cat(logits_list, dim=1)
    
    # 处理每个点可能被多个标签预测的情况
    # 找到每个点被预测为正的所有标签
    threshold = 0.5  # 大于0.5视为预测为该类别
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits)  # 将logits转换为概率
    # print('logits:', logits)
    # print('probs:', probs)
    
    # 对于每个点，确定其最终类别
    pred_labels = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
    for i in range(probs.shape[0]):
        # 找到所有预测概率大于阈值的标签
        positive_labels = torch.where(probs[i] > threshold)[0]
        
        if len(positive_labels) == 0:
            # 没有预测到任何标签，设为0（未标记）
            pred_labels[i] = 0
        elif len(positive_labels) == 1:
            # 只有一个标签，直接使用
            pred_labels[i] = positive_labels[0] + 1  # 假设标签从1开始
        else:
            # 多个标签，选择概率最高的类别
            max_prob = -float('inf')
            selected_label = 0
            for label in positive_labels:
                prob = probs[i, label].item()
                if prob > max_prob:
                    max_prob = prob
                    selected_label = label
            pred_labels[i] = selected_label + 1  # 假设标签从1开始
    
    # 上采样到完整点云
    xyz_full = xyz_full.squeeze()
    
    # 分配最近邻
    chunk_len = xyz_full.shape[0] // N_CHUNKS + 1
    closest_idx_list = []
    for i in range(N_CHUNKS):
        cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
        # 计算距离
        dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2  # [chunk_size, n_subsampled_pts, 3]
        cur_dist = (dist_all.sum(dim=-1))**0.5  # [chunk_size, n_subsampled_pts]
        min_idxs = torch.min(cur_dist, 1)[1]  # 找到每个点的最近邻
        del cur_dist
        closest_idx_list.append(min_idxs)
    all_nn_idxs = torch.cat(closest_idx_list, axis=0)
    
    # 获取完整点云的预测结果
    pred_full = pred_labels[all_nn_idxs].cpu()
    
    # 计算准确率
    acc = ((pred_full == gt_full) * 1).sum() / pred_full.shape[0]
    pred_np = pred_full.numpy()
    label_np = gt_full.squeeze().numpy()

    # 保存分割结果（保持原有逻辑）
    if savepath is not None:
        assert savepath.endswith('.ply')

        pngsavepath = savepath.replace('.ply', '.png')

        # 保存预测结果
        caption_list = visualize_pt_labels(xyz_visualization.cpu(), pred_full.cpu(), 
                                         save_path=savepath, save_rendered_path=pngsavepath)
        
        # 保存真实标签
        savepath_gt = savepath.replace(".ply", "_gt.ply") 
        pngsavepath_gt = pngsavepath.replace(".png", "_gt.png")
        caption_list = visualize_pt_labels(xyz_visualization.cpu(), gt_full.squeeze().cpu(), 
                                         save_path=savepath_gt, save_rendered_path=pngsavepath_gt)
        print('caption_list:', caption_list)
        
        # 保存特征
        featssavepath = savepath.replace("_gt.ply", "_feats.ply")
        save_feats_to_ply(backbone_feat, xyz_sub, featssavepath)
          
    # 计算每个类别的IoU
    part_ious = []
    for part in range(n_parts):
        # 注意：这里假设标签是从1开始的
        I = np.sum(np.logical_and(pred_np == part + 1, label_np == part + 1))
        U = np.sum(np.logical_or(pred_np == part + 1, label_np == part + 1))
        if U > 0:  # 避免除以零
            iou = I / float(U)
            part_ious.append(iou)
    
    # 计算平均IoU和准确率
    full_miou = np.mean(part_ious) if part_ious else 0.0
    full_macc = acc.item()
    
    return full_miou, full_macc

def evaluate3d(model, dataloader, panoptic=False, N_CHUNKS=1, visualize_seg=False, visualize_all_heatmap=False, savedir=None, testname=None): # evaluate loader can only have batch size=1
    temperature = np.exp(model.ln_logit_scale.item())
    iou_full_list = []

    ########### 获得feats 进行评价
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and "full" not in key:
                    data[key] = data[key].cuda(non_blocking=True)

            # model_output = model(x=data)
            data['mask_offset'] = torch.tensor(
                    [data['label_embeds'].shape[0]],  # 维度与 offset 一致（[1]）
                    device=data['offset'].device  # 对齐设备（此处为 cuda:0）
                )
            model_output = model(data)
            # 情况1：输出是元组或列表（多返回值，如网络6输出分割+bbox）
            if isinstance(model_output, (tuple, list)):
                if len(model_output) == 0:
                    raise ValueError("模型输出为空元组/列表，无法提取net_out")
                net_out = model_output[0]  # 取第一个元素
                # print(f"检测到元组/列表输出（共{len(model_output)}个结果），已取第1个作为net_out")
            # 情况3：输出是单个张量（如网络1/2仅输出seg_logits）
            elif isinstance(model_output, torch.Tensor):
                net_out = model_output
                # print("检测到单个张量输出，直接作为net_out")
            # 情况4：不支持的输出格式（提醒用户扩展）
            else:
                raise TypeError(
                    f"不支持的模型输出格式：{type(model_output)}，"
                    "请扩展get_first_net_out函数以支持该格式（当前支持：张量、元组、列表、字典）"
                )


            text_embeds = data['label_embeds']
            gt_full = data["gt_full"]
            xyz_sub = data["coord"]
            xyz_full = data["xyz_full"]
            cat = data["class_name"][0]
            

            # if savedir is not None:
            #     os.makedirs(os.path.join(savedir, cat), exist_ok=True)
            #     savepath = os.path.join(savedir, cat, str(i)+".ply")
            # else:
            #     savepath = None
            # try:
            #     # 限制保存分割结果的个数 ： 判断os.path.join(savedir, cat)是否有10个文件，如果有，则savepath = None
            #     if os.path.exists(os.path.join(savedir, cat)) and len(os.listdir(os.path.join(savedir, cat))) >= 10:
            #         savepath = None
            # except:
            #     print('限制保存分割结果的个数 fail')
            # uidname = data["uidname"][0]
            # savepath = f'/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/results_tmp/infer_objaverse/{uidname}.ply'
            savepath = None
            if testname == 'feats':
                full_miou, _ = compute_3d_iou_upsample(net_out, # n_subsampled_pts, feat_dim
                                                text_embeds, # n_parts, feat_dim
                                                temperature,
                                                cat,
                                                xyz_sub,
                                                xyz_full, # n_pts, 3
                                                gt_full, # n_pts,
                                                panoptic=panoptic,
                                                N_CHUNKS=N_CHUNKS,
                                                visualize_seg=visualize_seg,
                                                savepath = savepath,
                                                visualize_all_heatmap=visualize_all_heatmap,
                                                xyz_visualization = data["xyz_visualization"])
            else:
                full_miou, _ = computePre_3d_iou(model_output, data, cat, xyz_sub, xyz_full=xyz_full, gt_full=gt_full, N_CHUNKS=N_CHUNKS,
                                               visualize_seg=visualize_seg,
                                               savepath = savepath,
                                               visualize_all_heatmap=visualize_all_heatmap,
                                               xyz_visualization = data["xyz_visualization"])

            
            # print(cat, full_miou)
            iou_full_list += [full_miou]
    full_miou = np.mean(iou_full_list)
    return full_miou



def eval_category_shapenetpart(data_root, category, model, apply_rotation=False, subset=False, decorated=True, use_tuned_prompt=False, visualize_seg=False, visualize_all_heatmap=False, testname=None):
    test_data = EvalShapeNetPart(data_root, category, apply_rotation=apply_rotation, subset=subset, decorated=decorated, use_tuned_prompt=use_tuned_prompt)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False,
                             collate_fn=collate_fn, 
                             num_workers=0, 
                             drop_last=False)
    stime = time.time()
    full_miou = evaluate3d(model, test_loader, panoptic=True, N_CHUNKS=1, visualize_seg=visualize_seg, visualize_all_heatmap=visualize_all_heatmap, testname=testname)
    print(f"{category}: miou: {full_miou}")
    etime = time.time()
    print(etime-stime)
    return full_miou, etime-stime






def eval_shapenetpart(data_root, model, apply_rotation, subset, decorated, use_tuned_prompt, visualize_seg=False, save_path=None, testname=None):
    shapenetpart_categories = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
        'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
        'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
    full_mious = []
    time_all = 0
    for cat in shapenetpart_categories:
        full_miou, time_cur = eval_category_shapenetpart(data_root, cat, model, apply_rotation=apply_rotation,\
                         subset=subset, decorated=decorated, visualize_seg=visualize_seg, use_tuned_prompt = use_tuned_prompt, testname=testname)
        full_mious.append(full_miou)
        time_all += time_cur
    full_miou_avg = np.mean(full_mious)
    print(f"full miou {full_miou_avg}")
    print(f"time {time_all}")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(f"{'类别':<15} {'iou':<10}\n")
            f.write("-" * 25 + "\n")
            for category, miou in zip(shapenetpart_categories, full_mious):
                f.write(f"{category:<15} {miou:<10.4f}\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'平均值':<15} {full_miou_avg:<10.4f}\n")

       
if __name__ == '__main__':
    set_seed(123)
    parser = argparse.ArgumentParser(description="Please specify a benchmark name and evaluation configurations")
    parser.add_argument("--benchmark", required=True, type=str, help='The benchmark to evaluate on. Should be Objaverse, ShapeNetPart, or PartNetE')
    parser.add_argument("--data_root", required=True, type=str, help='Root directory of the benchmark data')
    parser.add_argument("--save_dir", required=True, type=str, help='results save to')
    parser.add_argument("--checkpoint_path", required=True, type=str, help='path of the checkpoint to evaluate')
    parser.add_argument("--net_type", required=True, type=str, help='使用的网络架构')
    parser.add_argument("--test_type", required=True, type=str, help='测试方式，feats or pre')
    parser.add_argument("--objaverse_split", type=str, help='If benchmark is Objaverse, specify "seenclass", "unseen" or "shapenetpart')
    parser.add_argument("--canonical", action='store_false', dest="rotate", help="whether to perform random rotation - this only applies to ShapeNetPart or PartNetE which have canonical orientations")
    parser.add_argument("--subset", action='store_true', dest="subsample", help="whether to evaluate on subset - this only applies to ShapeNetPart of PartNetE")
    parser.add_argument("--part_query", action='store_false', dest="decorate", help="if true, evaluate with {part} of a {object} as query prompt; if false, evaluate with {part} as query prompt")
    parser.add_argument("--use_shapenetpart_topk_prompt", action='store_true', help="This only applies to ShapeNetPart or Objaverse-ShapeNetPart. Whether to use the topk prompt following PointCLIPV2's procedures to choose prompts")
    parser.set_defaults(rotate=True, subsample=False, decorate=True, use_shapenetpart_topk_prompt=False)  # 这里设置，不进行旋转扰动
    args = parser.parse_args()
    


    if args.net_type == 'net1' or args.net_type == 'net2':
        model = load_model(args.checkpoint_path)
        # model = muti_load_model(args.checkpoint_path, net_type=args.net_type)
    elif args.net_type == 'net3':
        from release_pipeline3.stage1_semanspace.halfd3com_worot_aligncates_decoder.mixdecodernet import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == 'net4':
        from release_pipeline5ab.ab4_partfieldloss_sizeaug_decoder_canoncolor.mixdecodernet import PointSemSegWithDecoder_test as PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args) 
    elif args.net_type == 'net5':
        from release_pipeline5ab.ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign.mixdecodernet import PointSemSegWithDecoder_test as PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)   
    elif args.net_type == 'net6':
        from release_pipeline5ab.ab6_partfieldloss_sizeaug_decoder_bbox.mixdecodernet import PointSemSegWithbboxDecoder 
        model = PointSemSegWithbboxDecoder(args=args)
    elif args.net_type == 'net7': # ab1_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox
        from release_pipeline6.ab1_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox.mixdecodernet import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    elif args.net_type == 'net8':  # ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox
        # from release_pipeline6.ab1_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox.mixdecodernet import PointSemSegWithDecoder
        # from release_pipeline6.ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox.mixdecodernet import PointSemSegWithDecoder
        from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
        model = PointSemSegWithDecoder(args=args)
    
    # if args.test_type == 'feats':
    #     pretrained_checkpoint = torch.load(args.checkpoint_path)  # 原代码中是 args.checkpoint_path，与你参考示例的 args.pretrained_path 统一
    #     pretrained_state_dict = pretrained_checkpoint["model_state_dict"]  # 提取权重字典
    #     backbone_weights = {
    #         # 关键：移除参数名前缀 'backbone.'（因为 model.backbone 的参数名不含该前缀）
    #         k.replace('backbone.', ''): v  
    #         for k, v in pretrained_state_dict.items()
    #         if k.startswith('backbone.')  # 只保留 backbone 模块的参数，彻底排除 decoder
    #     }
    #     model.backbone.load_state_dict(backbone_weights)  # 
    # else:
    #     model.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"])
    model.eval()
    model = model.cuda()
    

    if args.benchmark == "ShapeNetPart":
        dataset='shapenetpart'
        netname = args.net_type
        testname = args.test_type
        print('args.rotate:', args.rotate)
        print('args.decorate:', args.decorate)
        save_pat = os.path.join(args.save_dir, f"{dataset}_{netname}_{testname}")
        if args.decorate:
            save_pat = save_pat + '_partcates_'
        if args.rotate:
            save_pat = save_pat + '_roted_'
        save_path = save_pat + '.txt'
        eval_shapenetpart(args.data_root, model, apply_rotation=args.rotate, subset=args.subsample, \
                decorated=args.decorate, use_tuned_prompt = args.use_shapenetpart_topk_prompt, \
                save_path=save_path, \
                testname=testname)

    
    