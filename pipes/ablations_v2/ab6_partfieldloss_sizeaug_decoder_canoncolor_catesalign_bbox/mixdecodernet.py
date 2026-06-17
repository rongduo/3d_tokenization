"""

 将之前得网络架构和新做的decoder 融合起来
"""

import torch
from dataclasses import asdict


from model.backbone.pt3.model import PointSemSeg  # 原backbone
# from release_module.decoder.ptstextnet import PointCloudTextTransformer, PointCloudTextTransformerConfig # decoder
# from release_module.decoder.ptstextnet import PointCloudTextCanoncolorTransformer, PointCloudTextCanoncolorTransformerConfig # decoder
from release_module.decoder.ptstextnet2 import PointCloudTextCanoncolorTransformer, PointCloudTextCanoncolorTransformerConfig # decoder 带bbox、canonical color


class PointSemSegWithDecoder(torch.nn.Module):
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
