"""
 python -m release_pipeline3.stage1_semanspace.halfd3com_worot_aligncates_decoder.mixdecoderloss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


############# 规范空间中的颜色loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class CanonicalColorLoss(nn.Module):
    def __init__(self):
        super(CanonicalColorLoss, self).__init__()

    def chamfer_distance(self, x, y):
        """
        计算两个点集之间的Chamfer距离
        x: [N, 3] 预测颜色点集
        y: [M, 3] 真实颜色点集
        """
        if x.size(0) == 0 or y.size(0) == 0:
            return torch.tensor(0.0, device=x.device)
            
        # 计算x中每个点到y中最近点的距离
        dist_matrix = torch.cdist(x, y, p=2)  # [N, M]
        min_dist_x = torch.min(dist_matrix, dim=1)[0].mean()  # 平均x到y的最小距离
        min_dist_y = torch.min(dist_matrix, dim=0)[0].mean()  # 平均y到x的最小距离
        
        return (min_dist_x + min_dist_y) / 2.0

    def forward(self, canoncolor_out, gt_color, pt_offset, mask_pts):
        """
        计算预测颜色与真实颜色的损失
        
        参数:
            canoncolor_out: [n_total_pts, 3] 预测的canonical颜色
            gt_color: [n_total_pts, 3] 真实的canonical颜色
            pt_offset: [BS+1,] 每个物体点云的起始索引
            mask_pts: list of [n_cur_masks, n_pts_cur_obj] 二进制掩码，指示点是否属于某个part
        """
        device = canoncolor_out.device
        batch_size = len(pt_offset) - 1  # 因为pt_offset是[BS+1,]
        total_loss = 0.0
        count = 0
        
        # 获取每个物体的起始索引
        obj_start_idxs = torch.cat((torch.tensor([0]).to(device), pt_offset[:-1]))
        
        # 遍历每个物体
        for obj_idx in range(batch_size):
            start_idx = obj_start_idxs[obj_idx]
            end_idx = pt_offset[obj_idx]
            obj_point_count = end_idx - start_idx
            
            if obj_point_count == 0:
                continue
                
            # 获取当前物体的掩码
            obj_mask_pts = mask_pts[obj_idx].to(device)  # [n_cur_masks, n_pts_cur_obj]
            n_masks = obj_mask_pts.size(0)
            
            # 遍历当前物体的每个part
            part_losses = []
            for mask_idx in range(n_masks):
                # 获取属于当前part的点索引
                part_mask = obj_mask_pts[mask_idx]  # [n_pts_cur_obj]
                part_point_indices = torch.nonzero(part_mask).squeeze(1)  # [n_pts_in_part]
                
                if part_point_indices.numel() < 2:  # 点太少无法计算有意义的距离
                    continue
                    
                # 转换为全局索引
                global_indices = start_idx + part_point_indices
                
                # 提取对应的预测颜色和真实颜色
                pred_colors = canoncolor_out[global_indices]  # [n_pts_in_part, 3]
                true_colors = gt_color[global_indices]       # [n_pts_in_part, 3]
                
                # 计算当前part的Chamfer距离
                part_dist = self.chamfer_distance(pred_colors, true_colors)
                part_losses.append(part_dist)
            
            # 计算当前物体的平均损失
            if part_losses:
                obj_loss = torch.mean(torch.stack(part_losses))
                total_loss += obj_loss
                count += 1
        
        # 计算整个批次的平均损失
        if count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / count








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
