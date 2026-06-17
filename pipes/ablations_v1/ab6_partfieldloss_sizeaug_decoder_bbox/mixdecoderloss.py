"""
 python -m release_pipeline3.stage1_semanspace.halfd3com_worot_aligncates_decoder.mixdecoderloss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


########  下面是bbox得loss函数
class PointBasedBBoxOffsetLoss(nn.Module):
    """
    适配毫米级误差的边界框损失函数
    1. 将误差从米（m）缩放为毫米（mm），避免小误差被平方过度压缩
    2. 调整Smooth L1的β值，适应毫米级误差范围
    3. 保持不加权平均，确保各部件贡献均等
    """
    def __init__(self, beta_mm: float = 10.0, debug: bool = False, min_point_count: int = 1):
        super().__init__()
        self.scale = 1000.0  # 米 -> 毫米（1m = 1000mm）
        self.beta = beta_mm  # 以毫米为单位的β值（默认10mm，超过此值用线性损失）
        self.debug = debug
        self.min_point_count = min_point_count

    def forward(
        self,
        bbox_pred: torch.Tensor,  # [total_parts, 6]
        pts: torch.Tensor,        # [total_pts, 3]
        pt_offset: torch.Tensor,  # [num_objects]
        mask_points: List[torch.Tensor]  # [num_objects]
    ) -> torch.Tensor:
        # 1. 初始验证
        num_objects = pt_offset.shape[0]
        if len(mask_points) != num_objects:
            raise ValueError(f"mask_points长度与物体数量不匹配: {len(mask_points)} vs {num_objects}")
        
        total_pred_parts = bbox_pred.shape[0]
        if total_pred_parts == 0:
            if self.debug:
                print("警告：没有预测部件，返回loss=0")
            return torch.tensor(0.0, device=bbox_pred.device)

        device = pts.device
        pt_offset = pt_offset.to(device)

        # 2. 预计算物体点云索引和中心点
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=pt_offset.dtype), pt_offset])
        obj_pt_indices = torch.arange(pts.shape[0], device=device)
        obj_ids = torch.searchsorted(point_offsets[1:], obj_pt_indices, right=False)
        
        ones = torch.ones_like(obj_pt_indices, dtype=torch.float32, device=device)
        obj_sums = torch.zeros((num_objects, 3), device=device)
        obj_counts = torch.zeros(num_objects, device=device)
        
        obj_sums.scatter_add_(0, obj_ids.unsqueeze(1).repeat(1, 3), pts)
        obj_counts.scatter_add_(0, obj_ids, ones)
        obj_counts = obj_counts.clamp(min=1.0)
        object_centers = obj_sums / obj_counts.unsqueeze(1)

        # 3. 处理部件，同步记录有效预测和真实值的索引
        gt_offsets_list = []
        valid_preds_list = []  # 存储有效预测值
        part_point_counts = []
        part_idx = 0  # 跟踪当前处理到的预测索引（关键）
        
        obj_pts_list = [pts[point_offsets[i]:point_offsets[i+1]] for i in range(num_objects)]
        
        for obj_idx in range(num_objects):
            obj_masks = mask_points[obj_idx].to(device)
            num_parts_in_obj = obj_masks.shape[0]  # 当前物体的部件总数
            if num_parts_in_obj == 0:
                if self.debug:
                    print(f"物体 {obj_idx} 没有部件，跳过")
                continue
            
            # 定位当前物体在bbox_pred中的预测片段（关键）
            pred_start = part_idx
            pred_end = part_idx + num_parts_in_obj
            if pred_end > total_pred_parts:
                raise ValueError(
                    f"物体 {obj_idx} 的预测索引超出范围: {pred_start}-{pred_end} > {total_pred_parts}"
                )
            obj_preds = bbox_pred[pred_start:pred_end]  # 当前物体的所有预测
            part_idx = pred_end  # 更新索引
            
            # 计算边界框和有效性
            obj_pts = obj_pts_list[obj_idx]
            obj_center = object_centers[obj_idx]
            part_mask_bool = (obj_masks > 0)
            point_counts = part_mask_bool.sum(dim=1)
            
            part_masks = part_mask_bool.unsqueeze(-1)
            masked_pts = obj_pts.unsqueeze(0) * part_masks
            part_min = torch.min(torch.where(part_masks, masked_pts, torch.inf), dim=1)[0]
            part_max = torch.max(torch.where(part_masks, masked_pts, -torch.inf), dim=1)[0]
            
            # 计算有效部件
            valid_parts = (point_counts >= self.min_point_count) & ~torch.any(torch.isinf(part_min), dim=1)
            valid_count = valid_parts.sum().item()
            invalid_count = num_parts_in_obj - valid_count
            
            if self.debug:
                print(f"物体 {obj_idx}: 总部件={num_parts_in_obj}, 有效={valid_count}, 无效={invalid_count}")
            
            if valid_count == 0:
                continue
            
            # 同步过滤：仅保留有效部件的真实值和预测值（核心修复）
            valid_gt = torch.cat([
                part_min[valid_parts] - obj_center,
                part_max[valid_parts] - obj_center
            ], dim=1)
            valid_pred = obj_preds[valid_parts]  # 关键：根据有效索引过滤预测值
            
            gt_offsets_list.append(valid_gt)
            valid_preds_list.append(valid_pred)  # 收集过滤后的预测值
            part_point_counts.append(point_counts[valid_parts])

        # 4. 处理空结果
        if not gt_offsets_list:
            if self.debug:
                print("警告：没有有效部件，返回loss=0")
            return torch.tensor(0.0, device=device)
        
        # 拼接过滤后的预测值和真实值（确保数量一致）
        gt_offsets = torch.cat(gt_offsets_list, dim=0)
        valid_bbox_pred = torch.cat(valid_preds_list, dim=0)  # 现在与gt_offsets数量匹配
        part_counts = torch.cat(part_point_counts, dim=0).float()

        # 最终验证（此时应匹配）
        if valid_bbox_pred.shape[0] != gt_offsets.shape[0]:
            raise ValueError(
                f"过滤后预测与真实部件数量仍不匹配: {valid_bbox_pred.shape[0]} vs {gt_offsets.shape[0]}"
            )
        if self.debug:
            print(f"\n过滤后总有效部件数: {gt_offsets.shape[0]}（预测与真实数量匹配）")

        # 5. 计算损失（使用过滤后的预测值）
        diff = (valid_bbox_pred - gt_offsets) * self.scale  # 毫米级误差
        abs_diff = torch.abs(diff)
        per_dim_loss = torch.where(
            abs_diff <= self.beta,
            0.5 * (abs_diff / self.beta) **2,
            (abs_diff / self.beta) - 0.5
        )
        
        per_part_loss = per_dim_loss.mean(dim=1)
        total_loss = per_part_loss.mean()

        # 调试打印（增加毫米级误差显示）
        if self.debug:
            print(f"\n===== 缩放后BBox Loss 调试信息 =====")
            print(f"总部件数量: {len(per_part_loss)}")
            print(f"整体平均loss: {total_loss.item():.6f}")
            print(f"（损失基准：10mm误差对应loss=0.5）")  # 便于直观理解
            
            # 误差分布（毫米级）
            print(f"\n毫米级误差分布:")
            print(f"  最小误差（平均）: {abs_diff.mean(dim=1).min().item():.2f} mm")
            print(f"  最大误差（平均）: {abs_diff.mean(dim=1).max().item():.2f} mm")
            
            # 前5个部件详情（显示毫米级误差）
            for i in range(min(5, len(per_part_loss))):
                print(f"\n部件 {i}:")
                print(f"  预测偏移量（米）: {bbox_pred[i].cpu().detach().numpy().round(6)}")
                print(f"  真实偏移量（米）: {gt_offsets[i].cpu().detach().numpy().round(6)}")
                print(f"  毫米级误差: {diff[i].cpu().detach().numpy().round(2)} mm")
                print(f"  部件损失: {per_part_loss[i].item():.6f}")
            print("==================================\n")

        return total_loss


######### 下面是分割得loss函数
class BalancedMaskCrossEntropyLoss(nn.Module):
    """
    基于点数量自动平衡的掩码交叉熵损失
    对包含点数量少的掩码自动赋予更高权重，解决类别不平衡问题
    无需手动设置mask_weights超参数
    """
    def __init__(self, label_smoothing: float = 0.1, weight_power: float = 1.0):
        super().__init__()
        self.label_smoothing = label_smoothing  # 标签平滑系数（0~1）
        self.weight_power = weight_power        # 权重幂次，控制平衡强度（建议1.0）
        self.eps = 1e-12                        # 数值稳定性常数

    def forward(
        self,
        decoder_out: torch.Tensor,  # 解码器输出：[total_pts, 1]
        mask_points: List[torch.Tensor],  # 多掩码列表：[batch_size]，每个元素为[M_i, N_i]
    ) -> torch.Tensor:
        """
        基于每个掩码的点数量自动计算权重，平衡不同大小的语义标签
        """
        # 1. 展平掩码并记录每个掩码的点数量
        mask_flat_list = []    # 存储展平后的掩码
        mask_sizes = []        # 存储每个掩码包含的点数量
        mask_indices = []      # 存储每个点属于哪个掩码的索引
        
        current_mask_id = 0    # 当前掩码的唯一标识
        for obj_masks in mask_points:
            M_i, N_i = obj_masks.shape  # M_i: 当前物体的掩码数量, N_i: 当前物体的点数量
            
            # 遍历当前物体的每个掩码
            for mask in obj_masks:
                # 展平单个掩码并添加到列表
                mask_flat = mask.view(-1)  # [N_i]
                mask_flat_list.append(mask_flat)
                
                # 记录当前掩码的点数量（只统计有效点）
                valid_points = torch.sum(mask_flat != 0).item()  # 非零点数量
                mask_sizes.append(valid_points if valid_points > 0 else 1)  # 避免零除
                
                # 记录每个点属于哪个掩码
                mask_indices.append(
                    torch.full((N_i,), current_mask_id, device=mask.device, dtype=torch.long)
                )
                
                current_mask_id += 1
        
        # 2. 拼接为全局张量
        mask_flat = torch.cat(mask_flat_list, dim=0).to(decoder_out.device)  # [total_pts]
        mask_indices = torch.cat(mask_indices, dim=0).to(decoder_out.device)  # [total_pts]
        total_masks = current_mask_id  # 掩码总数
        
        # 3. 验证维度一致性
        if mask_flat.shape[0] != decoder_out.shape[0]:
            raise ValueError(
                f"掩码展平后长度 {mask_flat.shape[0]} 与解码器输出长度 {decoder_out.shape[0]} 不匹配！"
            )

        # 4. 过滤有效点
        valid_mask = (mask_flat >= 0)  # 假设-1为无效值
        valid_pred = decoder_out[valid_mask].squeeze()  # [valid_pts]
        valid_label = mask_flat[valid_mask].float()      # [valid_pts]
        valid_mask_ids = mask_indices[valid_mask]       # [valid_pts]
        
        if valid_pred.numel() == 0:
            return torch.tensor(0.0, device=decoder_out.device)

        # 5. 应用标签平滑
        if self.label_smoothing > 0:
            valid_label = valid_label * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        # 6. 计算每个掩码的权重（核心改进）
        # 基于点数量的倒数计算权重，点越少权重越高
        mask_sizes_tensor = torch.tensor(mask_sizes, device=decoder_out.device, dtype=torch.float32)
        mask_weights = 1.0 / (mask_sizes_tensor ** self.weight_power + self.eps)
        
        # 归一化权重，确保总权重合理
        mask_weights = mask_weights / mask_weights.sum() * total_masks  # 权重和为掩码总数
        
        # 为每个有效点分配其所属掩码的权重
        point_weights = mask_weights[valid_mask_ids]

        # 7. 计算加权损失
        point_loss = F.binary_cross_entropy_with_logits(
            valid_pred,
            valid_label,
            reduction='none'  # 保留每个点的损失
        )
        
        # 加权平均：点少的掩码贡献更大
        weighted_loss = (point_loss * point_weights).sum() / point_weights.sum()

        return weighted_loss
    


'''import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class BinaryLossConfig:
    """二分类损失配置类（标签为1和-1）"""
    loss_type: str = "cross_entropy"
    mask_weights: Optional[List[float]] = None  # 多掩码权重
    batch_reduction: str = "mean"
    total_reduction: str = "mean"
    label_smoothing: float = 0.1  # 自定义标签平滑系数
    epsilon: float = 1e-6


class BinaryMultiMaskLoss(nn.Module):
    """修复掩码权重数量不匹配问题的二分类损失函数"""
    def __init__(self, cfg: BinaryLossConfig = None):
        super().__init__()
        self.cfg = cfg or BinaryLossConfig()
        self._validate_config()

    def _validate_config(self):
        """验证配置参数"""
        assert self.cfg.loss_type == "cross_entropy", "仅支持交叉熵损失"
        assert self.cfg.batch_reduction in ["mean", "sum"], "不支持的批次聚合方式"
        assert self.cfg.total_reduction in ["mean", "sum", "none"], "不支持的总聚合方式"
        assert 0 <= self.cfg.label_smoothing <= 1, "标签平滑系数需在[0,1]范围内"
        if self.cfg.mask_weights is not None:
            assert all(w >= 0 for w in self.cfg.mask_weights), "掩码权重不能为负"

    def _convert_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """将1/-1标签转换为0/1标签"""
        assert torch.all((labels == 1) | (labels == -1)), "标签必须仅包含1和-1"
        return (labels + 1) // 2  # 1→1, -1→0

    def _apply_label_smoothing(self, labels: torch.Tensor) -> torch.Tensor:
        """手动实现二分类标签平滑"""
        if self.cfg.label_smoothing == 0:
            return labels.float()
        return labels.float() * (1 - self.cfg.label_smoothing) + 0.5 * self.cfg.label_smoothing

    def _expand_labels(self, target: torch.Tensor, offset: torch.LongTensor, text_to_point_batch: torch.Tensor) -> torch.Tensor:
        """扩展标签并应用平滑"""
        converted_target = self._convert_labels(target)
        starts = torch.cat([torch.tensor([0], device=offset.device, dtype=offset.dtype), offset[:-1]])
        batch_sizes = offset - starts
        text_batch_sizes = batch_sizes[text_to_point_batch]

        expand_indices = []
        for i in range(len(text_batch_sizes)):
            batch_idx = text_to_point_batch[i].item()
            start = starts[batch_idx]
            end = offset[batch_idx]
            expand_indices.append(torch.arange(start, end, device=target.device))
        
        expanded = converted_target[torch.cat(expand_indices)]
        return self._apply_label_smoothing(expanded)

    def _expand_masks(self, mask_points: List[torch.Tensor], offset: torch.LongTensor, text_to_point_batch: torch.Tensor) -> torch.Tensor:
        """扩展掩码为全局张量"""
        device = offset.device
        total_texts = len(text_to_point_batch)
        max_num_masks = max(mask.shape[0] for mask in mask_points) if mask_points else 0
        
        expanded_masks = []
        for text_idx in range(total_texts):
            batch_idx = text_to_point_batch[text_idx].item()
            masks = mask_points[batch_idx]
            num_masks, num_points = masks.shape
            
            if num_masks < max_num_masks:
                pad = torch.zeros((max_num_masks - num_masks, num_points), dtype=torch.bool, device=device)
                masks = torch.cat([masks, pad], dim=0)
            
            expanded_masks.append(masks)

        return torch.cat(expanded_masks, dim=1) if expanded_masks else torch.empty(0, device=device)

    def _adjust_mask_weights(self, num_masks: int) -> List[float]:
        """
        自动调整掩码权重以匹配实际掩码数量
        如果提供的权重数量不足，则用平均权重填充剩余部分
        """
        if self.cfg.mask_weights is None:
            return [1.0 / num_masks] * num_masks if num_masks > 0 else []
        
        # 检查权重数量是否匹配
        if len(self.cfg.mask_weights) == num_masks:
            return self.cfg.mask_weights
        
        # 如果权重数量不足，用平均权重填充
        if len(self.cfg.mask_weights) < num_masks:
            remaining = num_masks - len(self.cfg.mask_weights)
            avg_weight = 1.0 / num_masks  # 剩余权重使用平均权重
            return self.cfg.mask_weights + [avg_weight] * remaining
        
        # 如果权重数量过多，截断到需要的数量
        return self.cfg.mask_weights[:num_masks]

    def forward(self, 
                pred: torch.Tensor,  # [sum(P×T), 1]
                target: torch.Tensor,  # [total_points] 1/-1
                offset: torch.LongTensor,
                new_offset: torch.LongTensor,
                mask_points: List[torch.Tensor],
                text_to_point_batch: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """修复掩码权重不匹配问题的前向传播"""
        # 1. 输入校验
        assert target.ndim == 1, f"标签必须为1D，当前{target.ndim}D"
        assert pred.ndim == 2 and pred.size(1) == 1, \
            f"二分类预测必须为2D[sum(P×T), 1]，当前形状{pred.shape}"
        total_pred_points = pred.size(0)
        total_texts = new_offset.size(0)

        # 2. 构建文本-点云映射
        if text_to_point_batch is None:
            text_to_point_batch = torch.arange(total_texts, device=offset.device)
        else:
            assert text_to_point_batch.size(0) == total_texts, "映射长度不匹配"

        # 3. 扩展标签和掩码
        expanded_target = self._expand_labels(target, offset, text_to_point_batch)
        expanded_masks = self._expand_masks(mask_points, offset, text_to_point_batch)
        num_masks = expanded_masks.size(0)

        # 4. 调整掩码权重以匹配实际掩码数量
        mask_weights = self._adjust_mask_weights(num_masks)
        
        # 5. 计算二分类交叉熵
        point_loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(1),  # [sum(P×T)]
            expanded_target,
            reduction="none"
        )

        # 6. 应用多掩码并加权
        if num_masks == 0:
            # 无掩码时直接计算整体损失
            if self.cfg.batch_reduction == "mean":
                return point_loss.mean()
            else:
                return point_loss.sum()

        mask_loss = expanded_masks.float() * point_loss.unsqueeze(0)

        # 7. 生成文本批次索引
        text_batch_indices = torch.zeros(total_pred_points, dtype=torch.long, device=pred.device)
        starts = torch.cat([torch.tensor([0], device=pred.device), new_offset[:-1]])
        for i in range(total_texts):
            text_batch_indices[starts[i]:new_offset[i]] = i

        # 8. 按文本批次聚合
        batch_mask_loss = []
        for mask_idx in range(num_masks):
            mask_sum = torch.zeros(total_texts, device=pred.device)
            mask_count = torch.zeros(total_texts, device=pred.device) + self.cfg.epsilon
            
            mask_sum.scatter_add_(0, text_batch_indices, mask_loss[mask_idx])
            mask_count.scatter_add_(0, text_batch_indices, expanded_masks[mask_idx].float())
            
            if self.cfg.batch_reduction == "mean":
                batch_loss = mask_sum / mask_count
            else:
                batch_loss = mask_sum
            
            batch_mask_loss.append(batch_loss * mask_weights[mask_idx])

        total_batch_loss = torch.stack(batch_mask_loss).sum(dim=0)

        # 9. 总Loss聚合
        if self.cfg.total_reduction == "mean":
            return total_batch_loss.mean()
        elif self.cfg.total_reduction == "sum":
            return total_batch_loss.sum()
        else:
            return total_batch_loss


# 示例用法
if __name__ == "__main__":
    # 模拟数据（包含3个掩码的情况）
    batch_size = 2
    text_per_batch = [2, 3]
    total_texts = sum(text_per_batch)
    
    point_counts = [1563, 2048]
    offset = torch.tensor([1563, 1563+2048], device="cuda")
    new_offset = torch.tensor([1563, 3126, 3126+2048, 3126+4096, 3126+6144], device="cuda")
    
    pred_logits = torch.randn(new_offset[-1], 1, device="cuda")
    target_labels = torch.where(
        torch.rand(offset[-1], device="cuda") > 0.5, 
        torch.tensor(1, device="cuda"), 
        torch.tensor(-1, device="cuda")
    )
    # 批次0有2个掩码，批次1有3个掩码（最大掩码数为3）
    mask_points = [
        torch.randn(2, 1563, device="cuda") > 0,
        torch.randn(3, 2048, device="cuda") > 0
    ]
    text_to_point_batch = torch.tensor([0,0,1,1,1], device="cuda")

    # 测试权重数量不足的情况（提供2个权重，自动补充第3个）
    loss_cfg = BinaryLossConfig(
        mask_weights=[1.0, 0.8],  # 实际需要3个权重，会自动补充
        label_smoothing=0.1
    )
    loss_fn = BinaryMultiMaskLoss(loss_cfg).cuda()

    total_loss = loss_fn(
        pred=pred_logits,
        target=target_labels,
        offset=offset,
        new_offset=new_offset,
        mask_points=mask_points,
        text_to_point_batch=text_to_point_batch
    )

    print(f"修复后总Loss: {total_loss.item():.4f}")
    print(f"实际掩码数量: 3, 使用的权重: {loss_fn._adjust_mask_weights(3)}")'''
