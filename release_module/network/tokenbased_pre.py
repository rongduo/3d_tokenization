"""

 将之前得网络架构和新做的decoder 融合起来
"""

import torch
import torch.nn as nn
from dataclasses import asdict


from model.backbone.pt3.model_token import PointSemSeg  # 原backbone
# from release_module.decoder.ptstextnet import PointCloudTextTransformer, PointCloudTextTransformerConfig # decoder
# from release_module.decoder.ptstextnet import PointCloudTextCanoncolorTransformer, PointCloudTextCanoncolorTransformerConfig # decoder
from release_module.decoder.ptstextnet2 import PointCloudTextCanoncolorWodecoderTransformer, PointCloudTextCanoncolorWodecoderTransformerConfig # decoder 带bbox、canonical color


class PointSemSegWithDecoder(torch.nn.Module):
    """整合backbone和decoder的完整模型"""
    def __init__(self, args):
        super().__init__()
        # 1. 初始化预训练的backbone（原PointSemSeg）
        self.backbone = PointSemSeg(args=args, dim_output=768)  # 保持原输出维度
        # 2. 初始化你的decoder
        cfg = PointCloudTextCanoncolorWodecoderTransformerConfig()
        cfg_dict = asdict(cfg)
        self.decoder = PointCloudTextCanoncolorWodecoderTransformer(cfg_dict)
        # 3. 保留原模型中的温度参数（如果需要）
        self.ln_logit_scale = self.backbone.ln_logit_scale  # 从backbone继承
        # 按 part token 对 Utonia 点特征 (1224-d) 做 mean/max 池化，再线性映射回 1224，写回 data['utonia_feat']（在 backbone 之前）
        self.utonia_token_feat_dim = 1224
        self.utonia_token_agg_proj = nn.Linear(
            2 * self.utonia_token_feat_dim, self.utonia_token_feat_dim
        )

    def _utonia_feat_token_pool(self, utonia_feat: torch.Tensor, data) -> torch.Tensor:
        """
        对每个 batch 样本、每个 token：对该 token 下所有点的 utonia_feat 做 mean 与 max，
        拼接后经线性层得到 part 级表征，并广播回该 token 下的每个点。
        utonia_feat: [N, 1224]；data['tokens']、[N]；data['offset']：点数累积边界。
        """
        if "tokens" not in data or data["tokens"] is None:
            return utonia_feat
        tokens = data["tokens"]
        if not isinstance(tokens, torch.Tensor) or tokens.numel() == 0:
            return utonia_feat
        if utonia_feat.shape[0] != tokens.shape[0]:
            return utonia_feat
        offset = data.get("offset")
        if offset is None:
            return utonia_feat
        if utonia_feat.shape[1] != self.utonia_token_feat_dim:
            return utonia_feat

        feat = utonia_feat
        tokens = tokens.to(device=feat.device, dtype=torch.long).reshape(-1)

        # 全 batch 一次分组：group_key = batch_id * base + token_id
        # token 非负时该编码可保证不同 batch 内同 token 不冲突。
        offset_t = offset.to(device=feat.device, dtype=torch.long).reshape(-1)
        if offset_t.numel() == 0:
            return utonia_feat
        n_points = feat.shape[0]
        batch_starts = torch.cat([offset_t.new_zeros(1), offset_t[:-1]], dim=0)
        counts = offset_t - batch_starts
        if int(counts.sum().item()) != n_points:
            return utonia_feat
        batch_ids = torch.repeat_interleave(
            torch.arange(offset_t.numel(), device=feat.device, dtype=torch.long),
            counts,
        )

        max_token = int(tokens.max().item())
        base = max_token + 1
        group_key = batch_ids * base + tokens

        # 压缩为连续组索引 [0, n_group)
        _, inv = torch.unique(group_key, sorted=False, return_inverse=True)
        n_group = int(inv.max().item()) + 1

        # group mean
        group_sum = torch.zeros((n_group, feat.shape[1]), device=feat.device, dtype=feat.dtype)
        group_sum.index_add_(0, inv, feat)
        group_cnt = torch.bincount(inv, minlength=n_group).to(feat.dtype).unsqueeze(1)
        group_mean = group_sum / group_cnt.clamp_min(1.0)

        # group max
        group_max = torch.full((n_group, feat.shape[1]), -torch.inf, device=feat.device, dtype=feat.dtype)
        group_max.scatter_reduce_(
            0,
            inv.unsqueeze(1).expand(-1, feat.shape[1]),
            feat,
            reduce="amax",
            include_self=True,
        )

        # [n_group, 2D] -> [n_group, D]，按 inv 广播回每个点
        group_pooled = self.utonia_token_agg_proj(torch.cat([group_mean, group_max], dim=1))
        return group_pooled[inv]

    def forward(self, data):
        # 0. 在进入 backbone 之前，按 token 聚合 Utonia 特征（mean+max -> Linear -> 1224）
        u = data.get("utonia_feat")
        if isinstance(u, torch.Tensor) and u.ndim == 2 and u.shape[1] == self.utonia_token_feat_dim:
            data["token_feat"] = self._utonia_feat_token_pool(u, data)

        
        # print('data.get("utonia_feat"):', data.get("utonia_feat").shape)
        # print('data["token_feat"]:', data["token_feat"].shape)
        # asdf


        # 1. backbone提取特征
        backbone_feat = self.backbone(data)  # 得到768维特征
        # return backbone_feat

        # 2. decoder处理特征
        decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset  = self.decoder(data['coord'], backbone_feat,
                                   data['label_embeds'], data['offset'], data['mask_offset'])  # 得到decoder输出
        
        # print('backbone_feat:', backbone_feat.shape)
        # print('data:', data.keys())
        # print('coord:', data['coord'].shape)
        # print('label_embeds:', data['label_embeds'].shape)
        # print('offset:', len(data['offset']))
        # print('mask_offset:', len(data['mask_offset']))
        # print('decoder_out:', decoder_out.shape)
        # print('decoder_out max:', torch.max(decoder_out))
        # print('decoder_out min:', torch.min(decoder_out))
        # print('canoncolor_out:', canoncolor_out.shape)
        # print('canoncolor max:', torch.max(canoncolor_out))
        # print('canoncolor min:', torch.min(canoncolor_out))
        # print('decoder_offset:', decoder_offset.shape)
        # print('decoder_out:', torch.max(decoder_out), torch.min(decoder_out))
        # asdf
        return backbone_feat, decoder_out, canoncolor_out, decoder_offset, bbox_pred, bbox_offset  # 返回中间特征和最终输出


class PointSemSegWithDecoder_test(torch.nn.Module):
    """整合backbone和decoder的完整模型"""
    def __init__(self, args):
        super().__init__()
        # 1. 初始化预训练的backbone（原PointSemSeg）
        self.backbone = PointSemSeg(args=args, dim_output=768)  # 保持原输出维度
        # 2. 初始化你的decoder
        cfg = PointCloudTextCanoncolorTransformerConfig()
        cfg_dict = asdict(cfg)
        self.decoder = PointCloudTextCanoncolorTransformer(cfg_dict)
        # 3. 保留原模型中的温度参数（如果需要）
        self.ln_logit_scale = self.backbone.ln_logit_scale  # 从backbone继承

    def forward(self, data):
        # 1. backbone提取特征
        backbone_feat = self.backbone(data)  # 得到768维特征
        # return backbone_feat
        
        # 2. decoder处理特征
        decoder_out, canoncolor_out, decoder_offset = self.decoder(data['coord'], backbone_feat,
                                   data['label_embeds'], data['offset'], data['mask_offset'])  # 得到decoder输出
        
        # print('backbone_feat:', backbone_feat.shape)
        # print('data:', data.keys())
        # print('coord:', data['coord'].shape)
        # print('label_embeds:', data['label_embeds'].shape)
        # print('offset:', len(data['offset']))
        # print('mask_offset:', len(data['mask_offset']))
        # print('decoder_out:', decoder_out.shape)
        # print('decoder_out max:', torch.max(decoder_out))
        # print('decoder_out min:', torch.min(decoder_out))
        # print('canoncolor_out:', canoncolor_out.shape)
        # print('canoncolor max:', torch.max(canoncolor_out))
        # print('canoncolor min:', torch.min(canoncolor_out))
        # print('decoder_offset:', decoder_offset.shape)
        # print('decoder_out:', torch.max(decoder_out), torch.min(decoder_out))
        # asdf
        return backbone_feat, decoder_out, decoder_offset, canoncolor_out  # 返回中间特征和最终输出

'''class PointSemSegWithDecoder(torch.nn.Module):
    """整合backbone和decoder的完整模型"""
    def __init__(self, args):
        super().__init__()
        # 1. 初始化预训练的backbone（原PointSemSeg）
        self.backbone = PointSemSeg(args=args, dim_output=768)  # 保持原输出维度
        # 2. 初始化你的decoder
        cfg = PointCloudTextTransformerConfig()
        cfg_dict = asdict(cfg)
        self.decoder = PointCloudTextTransformer(cfg_dict)
        # 3. 保留原模型中的温度参数（如果需要）
        self.ln_logit_scale = self.backbone.ln_logit_scale  # 从backbone继承

    def forward(self, data):
        # 1. backbone提取特征
        backbone_feat = self.backbone(data)  # 得到768维特征
        
        # 2. decoder处理特征
        decoder_out, decoder_offset = self.decoder(data['coord'], backbone_feat,
                                   data['label_embeds'], data['offset'], data['mask_offset'])  # 得到decoder输出
        
        # print('backbone_feat:', backbone_feat.shape)
        # print('data:', data.keys())
        # print('coord:', data['coord'].shape)
        # print('label_embeds:', data['label_embeds'].shape)
        # print('offset:', len(data['offset']))
        # print('mask_offset:', len(data['mask_offset']))
        # print('decoder_out:', decoder_out.shape)
        # print('decoder_offset:', decoder_offset.shape)
        # print('decoder_out:', torch.max(decoder_out), torch.min(decoder_out))
        # asdf
        return backbone_feat, decoder_out, decoder_offset  # 返回中间特征和最终输出'''
