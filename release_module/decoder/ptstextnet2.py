###这个是batch * data size  ， feats size 组织得，通过参数offset 和mask_offset 记录了每个batch结束得index ；不用遍历的加速版
### 进一步修改，使得 一个text embeding 对应一个output
from __future__ import annotations  # PEP 585 compat for Python < 3.9

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from einops import rearrange  # 移除repeat依赖，仅保留必要操作

from release_module.decoder.transformers.perceiver_1d import Perceiver
from release_module.decoder.transformers.attention import ResidualCrossAttentionBlock
from release_module.decoder.utils.checkpoint import checkpoint
from release_module.decoder.utils.base import BaseModule

from release_module.decoder.autoencoders.michelangelo_autoencoder import get_embedder





################### decoder 分割、 canonical color 、 bbox pre
@dataclass
class PointCloudTextCanoncolorTransformerConfig(BaseModule.Config):
    # 点云相关配置
    point_cloud_embed_type: str = "fourier"
    point_cloud_num_freqs: int = 8  # 6
    point_cloud_include_pi: bool = False  # True
    
    # 模型维度配置
    feature_dim: int = 768  # 点云和文本的特征维度
    width: int = 768        # 模型隐藏层维度
    heads: int = 12         # 注意力头数
    num_self_attn_layers: int = 16  # 6  # Self Attention层数   ????????
    
    # 注意力配置
    init_scale: float = 0.25
    qkv_bias: bool = False  # True
    use_flash: bool = True  # False
    use_checkpoint: bool = False
    
    # 输出配置
    output_dim: int = 1  # 预测输出维度（概率值）
    color_output_dim: int = 3  # 颜色预测输出维度（R, G, B）
    bbox_output_dim: int = 6  # 3D BBox预测输出维度 (x_min, y_min, z_min, x_max, y_max, z_max)


class PointCloudTextCanoncolorTransformer(BaseModule):
    Config = PointCloudTextCanoncolorTransformerConfig
    
    def __init__(self, cfg=None):
        super().__init__(cfg)
    
    def configure(self) -> None:
        super().configure()
        
        # 初始化点云坐标嵌入器
        self.pc_embedder = get_embedder(
            embed_type=self.cfg.point_cloud_embed_type,
            num_freqs=self.cfg.point_cloud_num_freqs,
            input_dim=3,
            include_pi=self.cfg.point_cloud_include_pi
        )
        
        # 点云特征投影层
        if hasattr(self.pc_embedder, 'out_dim'):
            embed_dim = self.pc_embedder.out_dim
        else:
            # 若嵌入器无out_dim，根据频率数计算（保持原逻辑）
            embed_dim = 3 * (2 ** self.cfg.point_cloud_num_freqs * 2)
        
        self.pc_feature_proj = nn.Linear(
            embed_dim + self.cfg.feature_dim, 
            self.cfg.width
        )
        
        # 文本特征投影层
        self.text_feature_proj = nn.Linear(
            self.cfg.feature_dim, 
            self.cfg.width
        )

        # 第一个Cross Attention
        self.first_cross_attn = ResidualCrossAttentionBlock(
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
        )
        
        # Self Attention层
        self.self_attn = Perceiver(
            n_ctx=None,
            width=self.cfg.width,
            layers=self.cfg.num_self_attn_layers,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )
        
        # 第二个Cross Attention
        self.second_cross_attn = ResidualCrossAttentionBlock(
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
        )
        
        # 输出投影层和LayerNorm（概率预测）
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        self.ln_post = nn.LayerNorm(self.cfg.width)
        
        # 颜色预测专用头（预测RGB三个通道）
        self.color_head = nn.Sequential(
            nn.LayerNorm(self.cfg.width),
            nn.Linear(self.cfg.width, self.cfg.width),
            nn.ReLU(),
            nn.Linear(self.cfg.width, 3)  # 直接预测RGB三个通道
        )
        
        # 3D BBox预测分支 - 用于提取点云整体特征
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化获取点云整体特征
        
        # 3D BBox预测分支 - 特征融合和预测层
        self.bbox_feature_fusion = nn.Linear(3 * self.cfg.width, self.cfg.width)
        self.bbox_proj1 = nn.Linear(self.cfg.width, self.cfg.width)
        self.bbox_act = nn.ReLU()
        self.bbox_proj2 = nn.Linear(self.cfg.width, self.cfg.bbox_output_dim)
    

    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (点云批次结束位置)
                mask_offset: torch.LongTensor  # shape: [batch_size] (文本组结束位置)
                ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:  
        # 返回：(概率输出, 颜色输出, 概率输出offset, 颜色输出offset, bbox预测, bbox offset)
        batch_size = offset.size(0)
        device = point_cloud.device
        dtype = offset.dtype
        
        # --------------------------
        # 1. 预处理：明确点云批次与文本组的对应关系
        # --------------------------
        # 点云偏移量（含起始0）：[0, p0_end, p1_end, ..., pB_end] → len=B+1
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), offset])
        # 文本组偏移量（含起始0）：[0, t0_end, t1_end, ..., tB_end] → len=B+1
        text_group_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), mask_offset])
        
        # 计算每个点云批次的长度：[p0_len, p1_len, ..., pB_len] → len=B
        point_batch_lens = point_offsets[1:] - point_offsets[:-1]
        # 计算每个文本组的文本数量：[t0_cnt, t1_cnt, ..., tB_cnt] → len=B
        text_group_cnts = text_group_offsets[1:] - text_group_offsets[:-1]
        
        # 处理空输入
        if batch_size == 0:
            return (torch.empty(0, self.cfg.output_dim, device=device),
                    torch.empty(0, 3, device=device),  # 颜色输出固定为3通道(RGB)
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, self.cfg.bbox_output_dim, device=device),
                    torch.empty(0, dtype=dtype, device=device))
        
        # --------------------------
        # 2. 生成“text-text所属点云批次”的映射关系
        # --------------------------
        # 示例：若点云批次0对应3个text，批次1对应2个text → text_to_point_batch = [0,0,0,1,1]
        text_to_point_batch = []
        for i in range(batch_size):
            text_to_point_batch.extend([i] * text_group_cnts[i].item())
        text_to_point_batch = torch.tensor(text_to_point_batch, device=device, dtype=dtype)  # [total_texts]
        total_texts = text_to_point_batch.size(0)
        
        # --------------------------
        # 3. 向量化准备：扩展点云数据以匹配text数量
        # --------------------------
        # 3.1 提取每个点云批次的原始数据
        point_batch_list = []  # 存储每个点云批次的 [p_len, 3] 数据
        point_feat_batch_list = []  # 存储每个点云批次的 [p_len, feat_dim] 特征
        for i in range(batch_size):
            p_start, p_end = point_offsets[i], point_offsets[i+1]
            point_batch_list.append(point_cloud[p_start:p_end])
            point_feat_batch_list.append(point_features[p_start:p_end])
        
        # 3.2 按text对应关系扩展点云数据：每个text对应其所属点云批次的完整数据
        # 输出形状：[total_texts, max_p_len, 3]（max_p_len为所有点云批次的最大长度）
        max_p_len = max([p.size(0) for p in point_batch_list]) if point_batch_list else 0
        expanded_point = torch.zeros(total_texts, max_p_len, 3, device=device, dtype=point_cloud.dtype)
        expanded_point_feat = torch.zeros(total_texts, max_p_len, point_features.size(1), 
                                        device=device, dtype=point_features.dtype)
        expanded_point_mask = torch.zeros(total_texts, max_p_len, device=device, dtype=torch.bool)  # 有效点掩码
        
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()  # 当前text所属的点云批次
            p_data = point_batch_list[point_batch_idx]  # [p_len, 3]
            p_feat = point_feat_batch_list[point_batch_idx]  # [p_len, feat_dim]
            p_len = p_data.size(0)
            
            # 填充当前text对应的点云数据
            expanded_point[text_idx, :p_len] = p_data
            expanded_point_feat[text_idx, :p_len] = p_feat
            expanded_point_mask[text_idx, :p_len] = True  # 标记有效点
        
        # --------------------------
        # 4. 文本特征处理：直接使用原始text特征（无需填充）
        # --------------------------
        # text_features已为[total_texts, text_feat_dim]，每个text对应一行
        text_proj = self.text_feature_proj(text_features)  # [total_texts, width]
        # 增加维度以匹配注意力输入（注意力通常期望[B, seq_len, dim]）
        text_proj = text_proj.unsqueeze(1)  # [total_texts, 1, width]（每个text视为长度1的序列）
        
        # --------------------------
        # 5. 点云特征处理
        # --------------------------
        # 点云嵌入：[total_texts, max_p_len, embed_dim]
        point_embedded = self.pc_embedder(expanded_point)
        # 拼接点云特征：[total_texts, max_p_len, embed_dim + feat_dim]
        point_feat = torch.cat([point_embedded, expanded_point_feat], dim=-1)
        # 点云投影：[total_texts, max_p_len, width]
        point_proj = self.pc_feature_proj(point_feat)
        # 掩码无效点：[total_texts, max_p_len, width]
        point_proj = point_proj * expanded_point_mask.unsqueeze(-1)
        
        # --------------------------
        # 6. 注意力计算（核心：每个text与对应点云交互）- 仅用于概率预测
        # --------------------------
        # 6.1 第一个交叉注意力：text（query）→ 点云（context）
        # 输入：query=[total_texts, 1, width], context=[total_texts, max_p_len, width]
        cross1_out = self.first_cross_attn(text_proj, point_proj)  # [total_texts, 1, width]
        # 掩码无效结果（此处text无无效值，可省略，但保持一致性）
        cross1_out = cross1_out * torch.ones_like(expanded_point_mask[:, :1], dtype=torch.float32).unsqueeze(-1)
        
        # 6.2 文本自注意力：[total_texts, 1, width] → 因text序列长度为1，自注意力无意义，可跳过或保留
        self_attn_out = self.self_attn(cross1_out)  # [total_texts, 1, width]
        
        # 6.3 第二个交叉注意力：点云（query）→ text（context）
        # 输入：query=[total_texts, max_p_len, width], context=[total_texts, 1, width]
        cross2_out = self.second_cross_attn(point_proj, self_attn_out)  # [total_texts, max_p_len, width]
        # 掩码无效点
        cross2_out = cross2_out * expanded_point_mask.unsqueeze(-1)
        
        # --------------------------
        # 7. 颜色预测分支（仅使用点云特征point_proj，不使用文本特征）
        # --------------------------
        # 7.1 按物体分组聚合点云特征
        object_point_feats = []
        for obj_idx in range(batch_size):
            # 找到属于当前物体的所有文本索引
            obj_text_mask = (text_to_point_batch == obj_idx)
            if obj_text_mask.sum() == 0:
                # 处理没有对应文本的物体
                obj_feat = torch.zeros(1, max_p_len, self.cfg.width, device=device)
            else:
                # 聚合该物体的所有点云投影特征（仅使用点云特征）
                obj_feat = point_proj[obj_text_mask].mean(dim=0, keepdim=True)  # [1, max_p_len, width]
            object_point_feats.append(obj_feat)
        object_point_feats = torch.cat(object_point_feats, dim=0)  # [batch_size, max_p_len, width]
        
        # 7.2 点级别颜色预测 - 预测RGB三个通道
        color_output_raw = self.color_head(object_point_feats)  # [batch_size, max_p_len, 3]
        color_output_raw = torch.sigmoid(color_output_raw)  # 将值归一化到0-1范围
        
        # 7.3 提取有效颜色预测（与点云一一对应）
        valid_color_output_list = []
        for obj_idx in range(batch_size):
            p_len = point_batch_lens[obj_idx].item()
            # 提取该物体对应的所有点的颜色
            obj_color_output = color_output_raw[obj_idx, :p_len]  # [p_len, 3]
            valid_color_output_list.append(obj_color_output)
        
        # 7.4 拼接所有有效颜色预测（总长度等于点云总点数）
        final_color_output = torch.cat(valid_color_output_list, dim=0)  # [total_points, 3]
        
        # 7.5 生成颜色输出的offset（与输入点云的offset保持一致）
        color_offset = offset  # 颜色输出与点云一一对应，所以offset相同
        
        # --------------------------
        # 8. 概率输出生成（使用文本-点云交互特征）
        # --------------------------
        # 8.1 投影到输出维度：概率预测 [total_texts, max_p_len, output_dim]
        prob_output_raw = self.output_proj(self.ln_post(cross2_out))
        
        # --------------------------
        # 9. 提取有效概率预测
        # --------------------------
        valid_prob_output_list = []
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()
            p_len = point_batch_lens[point_batch_idx].item()
            valid_prob_output = prob_output_raw[text_idx, :p_len]  # [p_len, output_dim]
            valid_prob_output_list.append(valid_prob_output)
        
        # 9.2 拼接所有有效概率预测
        final_prob_output = torch.cat(valid_prob_output_list, dim=0)  # [sum(p_len×text_cnt), output_dim]
        
        # --------------------------
        # 10. 生成概率输出的新offset
        # --------------------------
        # 计算每个组合的长度（即对应点云批次的长度）
        combo_lens = []
        for i in range(batch_size):
            combo_lens.extend([point_batch_lens[i].item()] * text_group_cnts[i].item())
        # 计算累积和作为新offset：[total_texts]
        prob_offset = torch.cumsum(torch.tensor(combo_lens, device=device, dtype=dtype), dim=0)
        
        # --------------------------
        # 11. 3D BBox预测分支：融合point_proj和cross1_out
        # --------------------------
        # 1. 整体物体特征：point_proj_global（全局池化，保留整体上下文）
        point_proj_global = self.global_pool(point_proj.transpose(1, 2)).squeeze(2)  # [total_texts, width]
        
        # 2. 部件特征：cross1_features（文本关注的部件区域）
        cross1_features = cross1_out.squeeze(1)  # [total_texts, width]
        
        # 3. 计算整体-部件的相对位置编码
        relative_pos_feat = cross1_features - point_proj_global  # [total_texts, width]
        
        # 4. 强化整体-部件关联：计算注意力权重
        part_vs_global_attn = torch.sigmoid(torch.bmm(
            cross1_features.unsqueeze(1),  # [total_texts, 1, width]
            point_proj_global.unsqueeze(2)  # [total_texts, width, 1]
        )).squeeze()  # [total_texts] → 部件与整体的关联度（0~1）
        
        # 5. 加权融合基础特征
        weighted_global = point_proj_global * part_vs_global_attn.unsqueeze(-1)  # [total_texts, width]
        weighted_part = cross1_features * (1 - part_vs_global_attn.unsqueeze(-1))  # [total_texts, width]

        # 6. 拼接融合：基础融合特征 + 相对位置编码
        fused_features = torch.cat([
            weighted_global, 
            weighted_part, 
            relative_pos_feat  # 相对位置特征
        ], dim=1)  # [total_texts, 3*width]
        
        # 7. 通过融合层统一特征维度
        fused_features = self.bbox_feature_fusion(fused_features)  # [total_texts, width]
        
        # 8. 预测部件BBox参数
        bbox_pred = self.bbox_proj2(self.bbox_act(self.bbox_proj1(fused_features)))
        
        # 8.5 生成BBox预测的offset
        bbox_offset = torch.arange(1, total_texts + 1, device=device, dtype=dtype) if total_texts > 0 else \
                      torch.empty(0, dtype=dtype, device=device)
        
        # --------------------------
        # 12. 验证一致性
        # --------------------------
        total_points = point_cloud.size(0)
        assert final_color_output.size(0) == total_points, \
            f"颜色输出总长度 {final_color_output.size(0)} 与点云总点数 {total_points} 不匹配"
        assert color_offset.size(0) == batch_size, \
            f"颜色输出offset长度 {color_offset.size(0)} 与批次大小 {batch_size} 不匹配"
        assert final_prob_output.size(0) == prob_offset[-1].item() if total_texts > 0 else 0, \
            f"概率输出总长度 {final_prob_output.size(0)} 与概率offset总长度 {prob_offset[-1].item() if total_texts>0 else 0} 不匹配"
        assert prob_offset.size(0) == total_texts, \
            f"概率输出offset长度 {prob_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_pred.size(0) == total_texts, \
            f"BBox预测数量 {bbox_pred.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_offset.size(0) == total_texts, \
            f"BBox offset长度 {bbox_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        
        return final_prob_output, final_color_output, prob_offset, bbox_pred, bbox_offset
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        if self.cfg.use_checkpoint and self.training:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)












################### canonical color 、 bbox pre  (去掉decoder 分割)
@dataclass
class PointCloudTextCanoncolorWodecoderTransformerConfig(BaseModule.Config):
    # 点云相关配置
    point_cloud_embed_type: str = "fourier"
    point_cloud_num_freqs: int = 8  # 6
    point_cloud_include_pi: bool = False  # True
    
    # 模型维度配置
    feature_dim: int = 768  # 点云和文本的特征维度
    width: int = 768        # 模型隐藏层维度
    heads: int = 12         # 注意力头数
    num_self_attn_layers: int = 16  # 6  # Self Attention层数   ????????
    
    # 注意力配置
    init_scale: float = 0.25
    qkv_bias: bool = False  # True
    use_flash: bool = True  # False
    use_checkpoint: bool = False
    
    # 输出配置
    output_dim: int = 1  # 预测输出维度（概率值）
    color_output_dim: int = 3  # 颜色预测输出维度（R, G, B）
    bbox_output_dim: int = 6  # 3D BBox预测输出维度 (x_min, y_min, z_min, x_max, y_max, z_max)


class PointCloudTextCanoncolorWodecoderTransformer(BaseModule):
    Config = PointCloudTextCanoncolorWodecoderTransformerConfig
    
    def __init__(self, cfg=None):
        super().__init__(cfg)
    
    def configure(self) -> None:
        super().configure()
        
        # 初始化点云坐标嵌入器
        self.pc_embedder = get_embedder(
            embed_type=self.cfg.point_cloud_embed_type,
            num_freqs=self.cfg.point_cloud_num_freqs,
            input_dim=3,
            include_pi=self.cfg.point_cloud_include_pi
        )
        
        # 点云特征投影层
        if hasattr(self.pc_embedder, 'out_dim'):
            embed_dim = self.pc_embedder.out_dim
        else:
            # 若嵌入器无out_dim，根据频率数计算（保持原逻辑）
            embed_dim = 3 * (2 ** self.cfg.point_cloud_num_freqs * 2)
        
        self.pc_feature_proj = nn.Linear(
            embed_dim + self.cfg.feature_dim, 
            self.cfg.width
        )
        
        # 文本特征投影层
        self.text_feature_proj = nn.Linear(
            self.cfg.feature_dim, 
            self.cfg.width
        )

        # 第一个Cross Attention
        self.first_cross_attn = ResidualCrossAttentionBlock(
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
        )
        
        # # Self Attention层
        # self.self_attn = Perceiver(
        #     n_ctx=None,
        #     width=self.cfg.width,
        #     layers=self.cfg.num_self_attn_layers,
        #     heads=self.cfg.heads,
        #     init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
        #     qkv_bias=self.cfg.qkv_bias,
        #     use_flash=self.cfg.use_flash,
        #     use_checkpoint=self.cfg.use_checkpoint
        # )
        
        # # 第二个Cross Attention
        # self.second_cross_attn = ResidualCrossAttentionBlock(
        #     width=self.cfg.width,
        #     heads=self.cfg.heads,
        #     init_scale=self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width),
        #     qkv_bias=self.cfg.qkv_bias,
        #     use_flash=self.cfg.use_flash,
        # )
        
        # 输出投影层和LayerNorm（概率预测）
        # self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        # self.ln_post = nn.LayerNorm(self.cfg.width)
        
        # 颜色预测专用头（预测RGB三个通道）
        self.color_head = nn.Sequential(
            nn.LayerNorm(self.cfg.width),
            nn.Linear(self.cfg.width, self.cfg.width),
            nn.ReLU(),
            nn.Linear(self.cfg.width, 3)  # 直接预测RGB三个通道
        )
        
        # 3D BBox预测分支 - 用于提取点云整体特征
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化获取点云整体特征
        
        # 3D BBox预测分支 - 特征融合和预测层
        self.bbox_feature_fusion = nn.Linear(3 * self.cfg.width, self.cfg.width)
        self.bbox_proj1 = nn.Linear(self.cfg.width, self.cfg.width)
        self.bbox_act = nn.ReLU()
        self.bbox_proj2 = nn.Linear(self.cfg.width, self.cfg.bbox_output_dim)
    

    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (点云批次结束位置)
                mask_offset: torch.LongTensor  # shape: [batch_size] (文本组结束位置)
                ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:  
        # 返回：(概率输出, 颜色输出, 概率输出offset, 颜色输出offset, bbox预测, bbox offset)
        batch_size = offset.size(0)
        device = point_cloud.device
        dtype = offset.dtype
        
        # --------------------------
        # 1. 预处理：明确点云批次与文本组的对应关系
        # --------------------------
        # 点云偏移量（含起始0）：[0, p0_end, p1_end, ..., pB_end] → len=B+1
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), offset])
        # 文本组偏移量（含起始0）：[0, t0_end, t1_end, ..., tB_end] → len=B+1
        text_group_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), mask_offset])
        
        # 计算每个点云批次的长度：[p0_len, p1_len, ..., pB_len] → len=B
        point_batch_lens = point_offsets[1:] - point_offsets[:-1]
        # 计算每个文本组的文本数量：[t0_cnt, t1_cnt, ..., tB_cnt] → len=B
        text_group_cnts = text_group_offsets[1:] - text_group_offsets[:-1]
        
        # 处理空输入
        if batch_size == 0:
            return (torch.empty(0, self.cfg.output_dim, device=device),
                    torch.empty(0, 3, device=device),  # 颜色输出固定为3通道(RGB)
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, self.cfg.bbox_output_dim, device=device),
                    torch.empty(0, dtype=dtype, device=device))
        
        # --------------------------
        # 2. 生成“text-text所属点云批次”的映射关系
        # --------------------------
        # 示例：若点云批次0对应3个text，批次1对应2个text → text_to_point_batch = [0,0,0,1,1]
        text_to_point_batch = []
        for i in range(batch_size):
            text_to_point_batch.extend([i] * text_group_cnts[i].item())
        text_to_point_batch = torch.tensor(text_to_point_batch, device=device, dtype=dtype)  # [total_texts]
        total_texts = text_to_point_batch.size(0)
        
        # --------------------------
        # 3. 向量化准备：扩展点云数据以匹配text数量
        # --------------------------
        # 3.1 提取每个点云批次的原始数据
        point_batch_list = []  # 存储每个点云批次的 [p_len, 3] 数据
        point_feat_batch_list = []  # 存储每个点云批次的 [p_len, feat_dim] 特征
        for i in range(batch_size):
            p_start, p_end = point_offsets[i], point_offsets[i+1]
            point_batch_list.append(point_cloud[p_start:p_end])
            point_feat_batch_list.append(point_features[p_start:p_end])
        
        # 3.2 按text对应关系扩展点云数据：每个text对应其所属点云批次的完整数据
        # 输出形状：[total_texts, max_p_len, 3]（max_p_len为所有点云批次的最大长度）
        max_p_len = max([p.size(0) for p in point_batch_list]) if point_batch_list else 0
        expanded_point = torch.zeros(total_texts, max_p_len, 3, device=device, dtype=point_cloud.dtype)
        expanded_point_feat = torch.zeros(total_texts, max_p_len, point_features.size(1), 
                                        device=device, dtype=point_features.dtype)
        expanded_point_mask = torch.zeros(total_texts, max_p_len, device=device, dtype=torch.bool)  # 有效点掩码
        
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()  # 当前text所属的点云批次
            p_data = point_batch_list[point_batch_idx]  # [p_len, 3]
            p_feat = point_feat_batch_list[point_batch_idx]  # [p_len, feat_dim]
            p_len = p_data.size(0)
            
            # 填充当前text对应的点云数据
            expanded_point[text_idx, :p_len] = p_data
            expanded_point_feat[text_idx, :p_len] = p_feat
            expanded_point_mask[text_idx, :p_len] = True  # 标记有效点
        
        # --------------------------
        # 4. 文本特征处理：直接使用原始text特征（无需填充）
        # --------------------------
        # text_features已为[total_texts, text_feat_dim]，每个text对应一行
        text_proj = self.text_feature_proj(text_features)  # [total_texts, width]
        # 增加维度以匹配注意力输入（注意力通常期望[B, seq_len, dim]）
        text_proj = text_proj.unsqueeze(1)  # [total_texts, 1, width]（每个text视为长度1的序列）
        
        # --------------------------
        # 5. 点云特征处理
        # --------------------------
        # 点云嵌入：[total_texts, max_p_len, embed_dim]
        point_embedded = self.pc_embedder(expanded_point)
        # 拼接点云特征：[total_texts, max_p_len, embed_dim + feat_dim]
        point_feat = torch.cat([point_embedded, expanded_point_feat], dim=-1)
        # 点云投影：[total_texts, max_p_len, width]
        point_proj = self.pc_feature_proj(point_feat)
        # 掩码无效点：[total_texts, max_p_len, width]
        point_proj = point_proj * expanded_point_mask.unsqueeze(-1)
        
        # --------------------------
        # 6. 注意力计算（核心：每个text与对应点云交互）- 仅用于概率预测
        # --------------------------
        # 6.1 第一个交叉注意力：text（query）→ 点云（context）
        # 输入：query=[total_texts, 1, width], context=[total_texts, max_p_len, width]
        cross1_out = self.first_cross_attn(text_proj, point_proj)  # [total_texts, 1, width]
        # 掩码无效结果（此处text无无效值，可省略，但保持一致性）
        cross1_out = cross1_out * torch.ones_like(expanded_point_mask[:, :1], dtype=torch.float32).unsqueeze(-1)
        
        # 6.2 文本自注意力：[total_texts, 1, width] → 因text序列长度为1，自注意力无意义，可跳过或保留
        # self_attn_out = self.self_attn(cross1_out)  # [total_texts, 1, width]
        
        # # 6.3 第二个交叉注意力：点云（query）→ text（context）
        # # 输入：query=[total_texts, max_p_len, width], context=[total_texts, 1, width]
        # cross2_out = self.second_cross_attn(point_proj, self_attn_out)  # [total_texts, max_p_len, width]
        # # 掩码无效点
        # cross2_out = cross2_out * expanded_point_mask.unsqueeze(-1)
        
        # --------------------------
        # 7. 颜色预测分支（仅使用点云特征point_proj，不使用文本特征）
        # --------------------------
        # 7.1 按物体分组聚合点云特征
        object_point_feats = []
        for obj_idx in range(batch_size):
            # 找到属于当前物体的所有文本索引
            obj_text_mask = (text_to_point_batch == obj_idx)
            if obj_text_mask.sum() == 0:
                # 处理没有对应文本的物体
                obj_feat = torch.zeros(1, max_p_len, self.cfg.width, device=device)
            else:
                # 聚合该物体的所有点云投影特征（仅使用点云特征）
                obj_feat = point_proj[obj_text_mask].mean(dim=0, keepdim=True)  # [1, max_p_len, width]
            object_point_feats.append(obj_feat)
        object_point_feats = torch.cat(object_point_feats, dim=0)  # [batch_size, max_p_len, width]
        
        # 7.2 点级别颜色预测 - 预测RGB三个通道
        color_output_raw = self.color_head(object_point_feats)  # [batch_size, max_p_len, 3]
        color_output_raw = torch.sigmoid(color_output_raw)  # 将值归一化到0-1范围
        
        # 7.3 提取有效颜色预测（与点云一一对应）
        valid_color_output_list = []
        for obj_idx in range(batch_size):
            p_len = point_batch_lens[obj_idx].item()
            # 提取该物体对应的所有点的颜色
            obj_color_output = color_output_raw[obj_idx, :p_len]  # [p_len, 3]
            valid_color_output_list.append(obj_color_output)
        
        # 7.4 拼接所有有效颜色预测（总长度等于点云总点数）
        final_color_output = torch.cat(valid_color_output_list, dim=0)  # [total_points, 3]
        
        # 7.5 生成颜色输出的offset（与输入点云的offset保持一致）
        color_offset = offset  # 颜色输出与点云一一对应，所以offset相同
        # --------------------------
        # 10. 生成概率输出的新offset
        # --------------------------
        # 计算每个组合的长度（即对应点云批次的长度）
        combo_lens = []
        for i in range(batch_size):
            combo_lens.extend([point_batch_lens[i].item()] * text_group_cnts[i].item())
        # 计算累积和作为新offset：[total_texts]
        prob_offset = torch.cumsum(torch.tensor(combo_lens, device=device, dtype=dtype), dim=0)
        
        # --------------------------
        # 11. 3D BBox预测分支：融合point_proj和cross1_out
        # --------------------------
        # 1. 整体物体特征：point_proj_global（全局池化，保留整体上下文）
        point_proj_global = self.global_pool(point_proj.transpose(1, 2)).squeeze(2)  # [total_texts, width]
        
        # 2. 部件特征：cross1_features（文本关注的部件区域）
        cross1_features = cross1_out.squeeze(1)  # [total_texts, width]
        
        # 3. 计算整体-部件的相对位置编码
        relative_pos_feat = cross1_features - point_proj_global  # [total_texts, width]
        
        # 4. 强化整体-部件关联：计算注意力权重
        part_vs_global_attn = torch.sigmoid(torch.bmm(
            cross1_features.unsqueeze(1),  # [total_texts, 1, width]
            point_proj_global.unsqueeze(2)  # [total_texts, width, 1]
        )).squeeze()  # [total_texts] → 部件与整体的关联度（0~1）
        
        # 5. 加权融合基础特征
        weighted_global = point_proj_global * part_vs_global_attn.unsqueeze(-1)  # [total_texts, width]
        weighted_part = cross1_features * (1 - part_vs_global_attn.unsqueeze(-1))  # [total_texts, width]

        # 6. 拼接融合：基础融合特征 + 相对位置编码
        fused_features = torch.cat([
            weighted_global, 
            weighted_part, 
            relative_pos_feat  # 相对位置特征
        ], dim=1)  # [total_texts, 3*width]
        
        # 7. 通过融合层统一特征维度
        fused_features = self.bbox_feature_fusion(fused_features)  # [total_texts, width]
        
        # 8. 预测部件BBox参数
        bbox_pred = self.bbox_proj2(self.bbox_act(self.bbox_proj1(fused_features)))
        
        # 8.5 生成BBox预测的offset
        bbox_offset = torch.arange(1, total_texts + 1, device=device, dtype=dtype) if total_texts > 0 else \
                      torch.empty(0, dtype=dtype, device=device)
        
        # --------------------------
        # 12. 验证一致性
        # --------------------------
        total_points = point_cloud.size(0)
        assert final_color_output.size(0) == total_points, \
            f"颜色输出总长度 {final_color_output.size(0)} 与点云总点数 {total_points} 不匹配"
        assert color_offset.size(0) == batch_size, \
            f"颜色输出offset长度 {color_offset.size(0)} 与批次大小 {batch_size} 不匹配"
        # assert final_prob_output.size(0) == prob_offset[-1].item() if total_texts > 0 else 0, \
        #     f"概率输出总长度 {final_prob_output.size(0)} 与概率offset总长度 {prob_offset[-1].item() if total_texts>0 else 0} 不匹配"
        assert prob_offset.size(0) == total_texts, \
            f"概率输出offset长度 {prob_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_pred.size(0) == total_texts, \
            f"BBox预测数量 {bbox_pred.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_offset.size(0) == total_texts, \
            f"BBox offset长度 {bbox_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        
        return 1, final_color_output, prob_offset, bbox_pred, bbox_offset
        # return 1, 1, prob_offset, bbox_pred, bbox_offset
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        if self.cfg.use_checkpoint and self.training:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
