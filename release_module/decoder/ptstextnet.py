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




############### 下面是decoder同时，加上bbox预测的分支
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from einops import rearrange

from release_module.decoder.transformers.perceiver_1d import Perceiver
from release_module.decoder.transformers.attention import ResidualCrossAttentionBlock
from release_module.decoder.utils.checkpoint import checkpoint
from release_module.decoder.utils.base import BaseModule

from release_module.decoder.autoencoders.michelangelo_autoencoder import get_embedder


################## 下面是同时预测分割点 ，bbox ，pose estimation的 decoder
@dataclass
class PointCloudTextBboxPoseestiTransformerConfig(BaseModule.Config):
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
    output_dim: int = 1  # 原始预测输出维度
    bbox_output_dim: int = 6  # 3D BBox预测输出维度 (x_min, y_min, z_min, x_max, y_max, z_max)


class PointCloudTextBboxPoseestiTransformer(BaseModule):
    Config = PointCloudTextBboxPoseestiTransformerConfig
    
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
        
        # 原始输出投影层和LayerNorm
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        # self.output_proj = nn.Sequential(
        #     nn.Linear(self.cfg.width, self.cfg.output_dim),
        #     nn.Sigmoid(),   
        #     nn.Linear(self.cfg.output_dim, self.cfg.output_dim),         
        # )
        self.ln_post = nn.LayerNorm(self.cfg.width)
        
        # 3D BBox预测分支 - 用于提取点云整体特征
        # self.global_pool = nn.AdaptiveAvgPool1d(self.cfg.width)  # 全局平均池化获取点云整体特征
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化获取点云整体特征
        
        # 3D BBox预测分支 - 特征融合和预测层
        self.bbox_feature_fusion = nn.Linear(3 * self.cfg.width, self.cfg.width)  # 修复此处
        self.bbox_proj1 = nn.Linear(self.cfg.width, self.cfg.width)
        self.bbox_act = nn.ReLU()
        self.bbox_proj2 = nn.Linear(self.cfg.width, self.cfg.bbox_output_dim)

        # 旋转估计分支 - 参考代码a的设置
        self.rot_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=self.cfg.width, nhead=4, batch_first=True),
            nn.ReLU(),
            nn.Linear(self.cfg.width, int(self.cfg.width/2)),
            nn.ReLU(),
            nn.Linear(int(self.cfg.width/2), 6)
            
        )
    
    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (点云批次结束位置)
                mask_offset: torch.LongTensor  # shape: [batch_size] (文本组结束位置)
                ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        # 返回值: (原始输出, 原始输出offset, bbox预测, bbox offset)
        batch_size = offset.size(0)
        device = point_cloud.device
        dtype = offset.dtype
        
        # 预处理：明确点云批次与文本组的对应关系
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), offset])
        text_group_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), mask_offset])
        
        point_batch_lens = point_offsets[1:] - point_offsets[:-1]
        text_group_cnts = text_group_offsets[1:] - text_group_offsets[:-1]
        
        # 处理空输入
        if batch_size == 0:
            return (torch.empty(0, self.cfg.output_dim, device=device), 
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, self.cfg.bbox_output_dim, device=device),
                    torch.empty(0, dtype=dtype, device=device))
        
        # 生成“text-text所属点云批次”的映射关系
        text_to_point_batch = []
        for i in range(batch_size):
            text_to_point_batch.extend([i] * text_group_cnts[i].item())
        text_to_point_batch = torch.tensor(text_to_point_batch, device=device, dtype=dtype)  # [total_texts]
        total_texts = text_to_point_batch.size(0)
        
        # 向量化准备：扩展点云数据以匹配text数量
        point_batch_list = []
        point_feat_batch_list = []
        for i in range(batch_size):
            p_start, p_end = point_offsets[i], point_offsets[i+1]
            point_batch_list.append(point_cloud[p_start:p_end])
            point_feat_batch_list.append(point_features[p_start:p_end])
        
        max_p_len = max([p.size(0) for p in point_batch_list]) if point_batch_list else 0
        expanded_point = torch.zeros(total_texts, max_p_len, 3, device=device, dtype=point_cloud.dtype)
        expanded_point_feat = torch.zeros(total_texts, max_p_len, point_features.size(1), 
                                        device=device, dtype=point_features.dtype)
        expanded_point_mask = torch.zeros(total_texts, max_p_len, device=device, dtype=torch.bool)
        
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()
            p_data = point_batch_list[point_batch_idx]
            p_feat = point_feat_batch_list[point_batch_idx]
            p_len = p_data.size(0)
            
            expanded_point[text_idx, :p_len] = p_data
            expanded_point_feat[text_idx, :p_len] = p_feat
            expanded_point_mask[text_idx, :p_len] = True
        
        # 文本特征处理
        text_proj = self.text_feature_proj(text_features)  # [total_texts, width]
        text_proj = text_proj.unsqueeze(1)  # [total_texts, 1, width]
        
        # 点云特征处理
        point_embedded = self.pc_embedder(expanded_point)  # [total_texts, max_p_len, embed_dim]
        point_feat = torch.cat([point_embedded, expanded_point_feat], dim=-1)  # [total_texts, max_p_len, embed_dim + feat_dim]
        point_proj = self.pc_feature_proj(point_feat)  # [total_texts, max_p_len, width]
        point_proj = point_proj * expanded_point_mask.unsqueeze(-1)  # 掩码无效点
        
        # 注意力计算
        # 第一个交叉注意力：text（query）→ 点云（context）
        cross1_out = self.first_cross_attn(text_proj, point_proj)  # [total_texts, 1, width]
        cross1_out = cross1_out * torch.ones_like(expanded_point_mask[:, :1], dtype=torch.float32).unsqueeze(-1)
        
        # 文本自注意力
        self_attn_out = self.self_attn(cross1_out)  # [total_texts, 1, width]
        
        # 第二个交叉注意力：点云（query）→ text（context）
        cross2_out = self.second_cross_attn(point_proj, self_attn_out)  # [total_texts, max_p_len, width]
        cross2_out = cross2_out * expanded_point_mask.unsqueeze(-1)
        
        # 原始输出生成
        output_raw = self.output_proj(self.ln_post(cross2_out))  # [total_texts, max_p_len, output_dim]
        
        valid_output_list = []
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()
            p_len = point_batch_lens[point_batch_idx].item()
            valid_output = output_raw[text_idx, :p_len]
            valid_output_list.append(valid_output)
        

        # # 这个有点问题，输出值域大概在 -1 ~1 ； 但是监督的信号是 0 ~ 1
        final_output = torch.cat(valid_output_list, dim=0) if valid_output_list else \
                      torch.empty(0, self.cfg.output_dim, device=device)  
        # final_output = torch.sigmoid(final_output)  # 任意值域 → 0-1 范围
        
        combo_lens = []
        for i in range(batch_size):
            combo_lens.extend([point_batch_lens[i].item()] * text_group_cnts[i].item())
        original_offset = torch.cumsum(torch.tensor(combo_lens, device=device, dtype=dtype), dim=0) \
                          if combo_lens else torch.empty(0, dtype=dtype, device=device)
        
        # --------------------------
        # 3D BBox预测分支：融合point_proj和cross1_out
        # --------------------------
        # 3D BBox预测分支（针对部件包围盒）
        # --------------------------
        # 1. 整体物体特征：point_proj_global（全局池化，保留整体上下文）
        point_proj_global = self.global_pool(point_proj.transpose(1, 2)).squeeze(2)  # [total_texts, width]
        
        # 2. 部件特征：cross1_features（文本关注的部件区域）
        cross1_features = cross1_out.squeeze(1)  # [total_texts, width]
        
        # 3. 计算整体-部件的相对位置编码（新增）
        # 特征差值反映部件相对于整体的“偏移特性”，增强空间关系建模
        relative_pos_feat = cross1_features - point_proj_global  # [total_texts, width]
        
        # 4. 强化整体-部件关联：计算注意力权重
        part_vs_global_attn = torch.sigmoid(torch.bmm(
            cross1_features.unsqueeze(1),  # [total_texts, 1, width]
            point_proj_global.unsqueeze(2)  # [total_texts, width, 1]
        )).squeeze()  # [total_texts] → 部件与整体的关联度（0~1）
        
        # 5. 加权融合基础特征
        weighted_global = point_proj_global * part_vs_global_attn.unsqueeze(1)  # [total_texts, width]
        weighted_part = cross1_features * (1 - part_vs_global_attn.unsqueeze(1))  # [total_texts, width]
        
        # 6. 拼接融合：基础融合特征 + 相对位置编码（核心修改）
        # 增加相对位置信息，帮助模型学习“部件在整体中的空间位置”
        fused_features = torch.cat([
            weighted_global, 
            weighted_part, 
            relative_pos_feat  # 新增：相对位置特征
        ], dim=1)  # [total_texts, 3*width]
        
        # 7. 通过融合层统一特征维度
        # 注意：需将bbox_feature_fusion的输入维度改为3*width
        fused_features = self.bbox_feature_fusion(fused_features)  # [total_texts, width]
        
        # 8. 预测部件BBox参数
        bbox_pred = self.bbox_proj2(self.bbox_act(self.bbox_proj1(fused_features)))
        
        # 8.5 生成BBox预测的offset
        bbox_offset = torch.arange(1, total_texts + 1, device=device, dtype=dtype) if total_texts > 0 else \
                      torch.empty(0, dtype=dtype, device=device)
        
        # --------------------------
        # 旋转估计分支（物体级）：使用point_proj作为输入
        # 对每个物体的所有文本特征取平均，得到物体级特征
        # --------------------------
        # 1. 按物体分组聚合特征
        object_feats = []
        for obj_idx in range(batch_size):
            # 找到属于当前物体的所有文本索引
            obj_text_mask = (text_to_point_batch == obj_idx)
            if obj_text_mask.sum() == 0:
                # 处理没有对应文本的物体
                obj_feat = torch.zeros(1, max_p_len, self.cfg.width, device=device)
            else:
                # 聚合该物体的所有点云投影特征
                obj_feat = point_proj[obj_text_mask].mean(dim=0, keepdim=True)  # [1, max_p_len, width]
            object_feats.append(obj_feat)
        object_feats = torch.cat(object_feats, dim=0)  # [batch_size, max_p_len, width]  # 2. 拼接所有物体特征
        rot_pred = self.rot_head(object_feats).mean(dim=1)  # [batch_size, 3] # 3. 物体级旋转预测
        
        # 验证一致性
        assert final_output.size(0) == original_offset[-1].item() if total_texts > 0 and original_offset.numel() > 0 else 0, \
            f"原始输出总长度 {final_output.size(0)} 与新offset总长度 {original_offset[-1].item() if (total_texts>0 and original_offset.numel()>0) else 0} 不匹配"
        assert original_offset.size(0) == total_texts, \
            f"原始输出offset长度 {original_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_pred.size(0) == total_texts, \
            f"BBox预测数量 {bbox_pred.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_offset.size(0) == total_texts, \
            f"BBox offset长度 {bbox_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        assert rot_pred.size(0) == batch_size, \
            f"旋转预测数量 {rot_pred.size(0)} 与物体数量 {batch_size} 不匹配"
        assert rot_pred.size(1) == 6, \
            f"旋转预测维度 {rot_pred.size(1)} 应为3"
        
        return final_output, original_offset, bbox_pred, bbox_offset, rot_pred
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        if self.cfg.use_checkpoint and self.training:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)



################ 下面是同时预测分割点， 和 bbox得 deocder
@dataclass
class PointCloudTextBboxTransformerConfig(BaseModule.Config):
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
    output_dim: int = 1  # 原始预测输出维度
    bbox_output_dim: int = 6  # 3D BBox预测输出维度 (x_min, y_min, z_min, x_max, y_max, z_max)


class PointCloudTextBboxTransformer(BaseModule):
    Config = PointCloudTextBboxTransformerConfig
    
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
        
        # 原始输出投影层和LayerNorm
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        # self.output_proj = nn.Sequential(
        #     nn.Linear(self.cfg.width, self.cfg.output_dim),
        #     nn.Sigmoid(),   
        #     nn.Linear(self.cfg.output_dim, self.cfg.output_dim),         
        # )
        self.ln_post = nn.LayerNorm(self.cfg.width)
        
        # 3D BBox预测分支 - 用于提取点云整体特征
        # self.global_pool = nn.AdaptiveAvgPool1d(self.cfg.width)  # 全局平均池化获取点云整体特征
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化获取点云整体特征
        
        # 3D BBox预测分支 - 特征融合和预测层
        self.bbox_feature_fusion = nn.Linear(3 * self.cfg.width, self.cfg.width)  # 修复此处
        self.bbox_proj1 = nn.Linear(self.cfg.width, self.cfg.width)
        self.bbox_act = nn.ReLU()
        self.bbox_proj2 = nn.Linear(self.cfg.width, self.cfg.bbox_output_dim)


        
    
    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (点云批次结束位置)
                mask_offset: torch.LongTensor  # shape: [batch_size] (文本组结束位置)
                ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        # 返回值: (原始输出, 原始输出offset, bbox预测, bbox offset)
        batch_size = offset.size(0)
        device = point_cloud.device
        dtype = offset.dtype
        
        # 预处理：明确点云批次与文本组的对应关系
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), offset])
        text_group_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), mask_offset])
        
        point_batch_lens = point_offsets[1:] - point_offsets[:-1]
        text_group_cnts = text_group_offsets[1:] - text_group_offsets[:-1]
        
        # 处理空输入
        if batch_size == 0:
            return (torch.empty(0, self.cfg.output_dim, device=device), 
                    torch.empty(0, dtype=dtype, device=device),
                    torch.empty(0, self.cfg.bbox_output_dim, device=device),
                    torch.empty(0, dtype=dtype, device=device))
        
        # 生成“text-text所属点云批次”的映射关系
        text_to_point_batch = []
        for i in range(batch_size):
            text_to_point_batch.extend([i] * text_group_cnts[i].item())
        text_to_point_batch = torch.tensor(text_to_point_batch, device=device, dtype=dtype)  # [total_texts]
        total_texts = text_to_point_batch.size(0)
        
        # 向量化准备：扩展点云数据以匹配text数量
        point_batch_list = []
        point_feat_batch_list = []
        for i in range(batch_size):
            p_start, p_end = point_offsets[i], point_offsets[i+1]
            point_batch_list.append(point_cloud[p_start:p_end])
            point_feat_batch_list.append(point_features[p_start:p_end])
        
        max_p_len = max([p.size(0) for p in point_batch_list]) if point_batch_list else 0
        expanded_point = torch.zeros(total_texts, max_p_len, 3, device=device, dtype=point_cloud.dtype)
        expanded_point_feat = torch.zeros(total_texts, max_p_len, point_features.size(1), 
                                        device=device, dtype=point_features.dtype)
        expanded_point_mask = torch.zeros(total_texts, max_p_len, device=device, dtype=torch.bool)
        
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()
            p_data = point_batch_list[point_batch_idx]
            p_feat = point_feat_batch_list[point_batch_idx]
            p_len = p_data.size(0)
            
            expanded_point[text_idx, :p_len] = p_data
            expanded_point_feat[text_idx, :p_len] = p_feat
            expanded_point_mask[text_idx, :p_len] = True
        
        # 文本特征处理
        text_proj = self.text_feature_proj(text_features)  # [total_texts, width]
        text_proj = text_proj.unsqueeze(1)  # [total_texts, 1, width]
        
        # 点云特征处理
        point_embedded = self.pc_embedder(expanded_point)  # [total_texts, max_p_len, embed_dim]
        point_feat = torch.cat([point_embedded, expanded_point_feat], dim=-1)  # [total_texts, max_p_len, embed_dim + feat_dim]
        point_proj = self.pc_feature_proj(point_feat)  # [total_texts, max_p_len, width]
        point_proj = point_proj * expanded_point_mask.unsqueeze(-1)  # 掩码无效点
        
        # 注意力计算
        # 第一个交叉注意力：text（query）→ 点云（context）
        cross1_out = self.first_cross_attn(text_proj, point_proj)  # [total_texts, 1, width]
        cross1_out = cross1_out * torch.ones_like(expanded_point_mask[:, :1], dtype=torch.float32).unsqueeze(-1)
        
        # 文本自注意力
        self_attn_out = self.self_attn(cross1_out)  # [total_texts, 1, width]
        
        # 第二个交叉注意力：点云（query）→ text（context）
        cross2_out = self.second_cross_attn(point_proj, self_attn_out)  # [total_texts, max_p_len, width]
        cross2_out = cross2_out * expanded_point_mask.unsqueeze(-1)
        
        # 原始输出生成
        output_raw = self.output_proj(self.ln_post(cross2_out))  # [total_texts, max_p_len, output_dim]
        
        valid_output_list = []
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()
            p_len = point_batch_lens[point_batch_idx].item()
            valid_output = output_raw[text_idx, :p_len]
            valid_output_list.append(valid_output)
        
        final_output = torch.cat(valid_output_list, dim=0) if valid_output_list else \
                      torch.empty(0, self.cfg.output_dim, device=device)
        # final_output = torch.sigmoid(final_output)  # 任意值域 → 0-1 范围
        
        combo_lens = []
        for i in range(batch_size):
            combo_lens.extend([point_batch_lens[i].item()] * text_group_cnts[i].item())
        original_offset = torch.cumsum(torch.tensor(combo_lens, device=device, dtype=dtype), dim=0) \
                          if combo_lens else torch.empty(0, dtype=dtype, device=device)
        
        # --------------------------
        # 3D BBox预测分支：融合point_proj和cross1_out
        # --------------------------
        # 3D BBox预测分支（针对部件包围盒）
        # --------------------------
        # 1. 整体物体特征：point_proj_global（全局池化，保留整体上下文）
        point_proj_global = self.global_pool(point_proj.transpose(1, 2)).squeeze(2)  # [total_texts, width]
        
        # 2. 部件特征：cross1_features（文本关注的部件区域）
        cross1_features = cross1_out.squeeze(1)  # [total_texts, width]
        
        # 3. 计算整体-部件的相对位置编码（新增）
        # 特征差值反映部件相对于整体的“偏移特性”，增强空间关系建模
        relative_pos_feat = cross1_features - point_proj_global  # [total_texts, width]
        
        # 4. 强化整体-部件关联：计算注意力权重
        part_vs_global_attn = torch.sigmoid(torch.bmm(
            cross1_features.unsqueeze(1),  # [total_texts, 1, width]
            point_proj_global.unsqueeze(2)  # [total_texts, width, 1]
        )).squeeze()  # [total_texts] → 部件与整体的关联度（0~1）
        
        # 5. 加权融合基础特征
        # weighted_global = point_proj_global * part_vs_global_attn.unsqueeze(1)  # [total_texts, width]
        # weighted_part = cross1_features * (1 - part_vs_global_attn.unsqueeze(1))  # [total_texts, width]
        ### 改成这样才能适配测试
        weighted_global = point_proj_global * part_vs_global_attn.unsqueeze(-1)  # 最后一个维度扩展
        weighted_part = cross1_features * (1 - part_vs_global_attn.unsqueeze(-1))   

        # 6. 拼接融合：基础融合特征 + 相对位置编码（核心修改）
        # 增加相对位置信息，帮助模型学习“部件在整体中的空间位置”
        fused_features = torch.cat([
            weighted_global, 
            weighted_part, 
            relative_pos_feat  # 新增：相对位置特征
        ], dim=1)  # [total_texts, 3*width]
        
        # 7. 通过融合层统一特征维度
        # 注意：需将bbox_feature_fusion的输入维度改为3*width
        fused_features = self.bbox_feature_fusion(fused_features)  # [total_texts, width]
        
        # 8. 预测部件BBox参数
        bbox_pred = self.bbox_proj2(self.bbox_act(self.bbox_proj1(fused_features)))
        
        # 8.5 生成BBox预测的offset
        bbox_offset = torch.arange(1, total_texts + 1, device=device, dtype=dtype) if total_texts > 0 else \
                      torch.empty(0, dtype=dtype, device=device)

        
        
        # 验证一致性
        assert final_output.size(0) == original_offset[-1].item() if total_texts > 0 and original_offset.numel() > 0 else 0, \
            f"原始输出总长度 {final_output.size(0)} 与新offset总长度 {original_offset[-1].item() if (total_texts>0 and original_offset.numel()>0) else 0} 不匹配"
        assert original_offset.size(0) == total_texts, \
            f"原始输出offset长度 {original_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_pred.size(0) == total_texts, \
            f"BBox预测数量 {bbox_pred.size(0)} 与文本总数 {total_texts} 不匹配"
        assert bbox_offset.size(0) == total_texts, \
            f"BBox offset长度 {bbox_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        
        
        return final_output, original_offset, bbox_pred, bbox_offset
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        if self.cfg.use_checkpoint and self.training:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)


######################## 下面是 进行分割 和 canonical color 预测的分支
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
    

    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (点云批次结束位置)
                mask_offset: torch.LongTensor  # shape: [batch_size] (文本组结束位置)
                ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:  
        # 返回：(概率输出, 颜色输出, 概率输出offset, 颜色输出offset)
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
        max_p_len = max([p.size(0) for p in point_batch_list])
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
        # 11. 验证一致性
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
        # print('prob_offset:', prob_offset.shape)
        # print('color_offset:', color_offset.shape)
        # print('prob_offset:', prob_offset)
        # print('color_offset:', color_offset)
        return final_prob_output, final_color_output, prob_offset # , color_offset
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        # # 注意：checkpoint不支持返回多个值，若启用需修改checkpoint逻辑或禁用
        # if self.cfg.use_checkpoint and self.training:
        #     # 临时禁用checkpoint以支持多返回值（或修改checkpoint适配tuple）
        #     # return checkpoint(self._forward, (point_cloud, ...), self.parameters(), self.cfg.use_checkpoint)
        #     return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        # else:
        #     return self._forward(point_cloud, point_features, text_features, offset, mask_offset)


################## 下面是仅进行分割的decoder分支

@dataclass
class PointCloudTextTransformerConfig(BaseModule.Config):
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
    output_dim: int = 1  # 预测输出维度


class PointCloudTextTransformer(BaseModule):
    Config = PointCloudTextTransformerConfig
    
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
        
        # 输出投影层和LayerNorm
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        self.ln_post = nn.LayerNorm(self.cfg.width)
    

    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (点云批次结束位置)
                mask_offset: torch.LongTensor  # shape: [batch_size] (文本组结束位置)
                ) -> tuple[torch.FloatTensor, torch.LongTensor]:  # (输出, 新offset)
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
            return torch.empty(0, self.cfg.output_dim, device=device), torch.empty(0, dtype=dtype, device=device)
        
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
        max_p_len = max([p.size(0) for p in point_batch_list])
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
        # 6. 注意力计算（核心：每个text与对应点云交互）
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
        # 7. 输出生成：每个text对应一个点云预测结果
        # --------------------------
        # 7.1 投影到输出维度：[total_texts, max_p_len, output_dim]
        output_raw = self.output_proj(self.ln_post(cross2_out))
        # 7.2 提取有效预测（只保留每个text对应点云批次的有效点）
        valid_output_list = []
        for text_idx in range(total_texts):
            point_batch_idx = text_to_point_batch[text_idx].item()
            p_len = point_batch_lens[point_batch_idx].item()
            # 提取当前text对应点云批次的有效预测：[p_len, output_dim]
            valid_output = output_raw[text_idx, :p_len]
            valid_output_list.append(valid_output)
        
        # 7.3 拼接所有有效预测：[sum(p_len×text_cnt), output_dim]
        final_output = torch.cat(valid_output_list, dim=0)
        
        # --------------------------
        # 8. 生成新offset：标记每个“text-点云”组合的结束位置
        # --------------------------
        # 计算每个组合的长度（即对应点云批次的长度）
        combo_lens = []
        for i in range(batch_size):
            combo_lens.extend([point_batch_lens[i].item()] * text_group_cnts[i].item())
        # 计算累积和作为新offset：[total_texts]
        new_offset = torch.cumsum(torch.tensor(combo_lens, device=device, dtype=dtype), dim=0)
        
        # --------------------------
        # 9. 验证一致性
        # --------------------------
        assert final_output.size(0) == new_offset[-1].item() if total_texts > 0 else 0, \
            f"输出总长度 {final_output.size(0)} 与新offset总长度 {new_offset[-1].item() if total_texts>0 else 0} 不匹配"
        assert new_offset.size(0) == total_texts, \
            f"新offset长度 {new_offset.size(0)} 与文本总数 {total_texts} 不匹配"
        
        return final_output, new_offset
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # 注意：checkpoint不支持返回多个值，若启用需修改checkpoint逻辑或禁用
        if self.cfg.use_checkpoint and self.training:
            # 临时禁用checkpoint以支持多返回值（或修改checkpoint适配tuple）
            # return checkpoint(self._forward, (point_cloud, ...), self.parameters(), self.cfg.use_checkpoint)
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)
    







'''############ 这个是batch * data size  ， feats size 组织得，通过参数offset 和mask_offset 记录了每个batch结束得index ；不用遍历的加速版
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from einops import repeat, rearrange

from release_module.decoder.transformers.perceiver_1d import Perceiver
from release_module.decoder.transformers.attention import ResidualCrossAttentionBlock
from release_module.decoder.utils.checkpoint import checkpoint
from release_module.decoder.utils.base import BaseModule

from release_module.decoder.autoencoders.michelangelo_autoencoder import get_embedder


@dataclass
class PointCloudTextTransformerConfig(BaseModule.Config):
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
    output_dim: int = 1  # 预测输出维度


class PointCloudTextTransformer(BaseModule):
    Config = PointCloudTextTransformerConfig
    
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
        
        # 输出投影层和LayerNorm
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        self.ln_post = nn.LayerNorm(self.cfg.width)
    
    def _forward(self, 
                point_cloud: torch.FloatTensor,  # shape: [total_points, 3]
                point_features: torch.FloatTensor,  # shape: [total_points, feat_dim]
                text_features: torch.FloatTensor,  # shape: [total_texts, text_feat_dim]
                offset: torch.LongTensor,  # shape: [batch_size] (每个批次的结束位置)
                mask_offset: torch.LongTensor  # shape: [batch_size] (每个文本批次的结束位置)
                ) -> torch.FloatTensor:
        batch_size = offset.size(0)
        device = point_cloud.device
        dtype = offset.dtype  # 保持索引数据类型一致
        
        # --------------------------
        # 1. 计算完整偏移量数组（修复维度不匹配）
        # --------------------------
        # 点云偏移量：[0, offset_0, offset_1, ..., offset_{batch-1}] → 长度batch_size+1
        point_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), offset])
        # 文本偏移量：同上
        text_offsets = torch.cat([torch.tensor([0], device=device, dtype=dtype), mask_offset])
        
        # 计算每个批次的有效数据量（长度均为batch_size）
        point_counts = point_offsets[1:] - point_offsets[:-1]
        text_counts = text_offsets[1:] - text_offsets[:-1]
        
        # 处理空批次情况
        if batch_size == 0:
            return torch.zeros(point_cloud.size(0), self.cfg.output_dim, device=device)
        
        max_points = point_counts.max()
        max_texts = text_counts.max()

        # --------------------------
        # 2. 生成批次索引与掩码（避免无效索引）
        # --------------------------
        # 点云索引：[batch_size, max_points]
        point_idx = torch.arange(max_points, device=device).unsqueeze(0).repeat(batch_size, 1)
        point_mask = point_idx < point_counts.unsqueeze(1)  # 有效点云掩码：True=有效
        
        # 计算全局索引（无效位置设为-1，避免越界）
        point_flat_indices = point_offsets[:-1].unsqueeze(1) + point_idx
        point_flat_indices = torch.where(point_mask, point_flat_indices, torch.tensor(-1, device=device, dtype=dtype))

        # 文本索引：[batch_size, max_texts]
        text_idx = torch.arange(max_texts, device=device).unsqueeze(0).repeat(batch_size, 1)
        text_mask = text_idx < text_counts.unsqueeze(1)  # 有效文本掩码：True=有效
        text_flat_indices = text_offsets[:-1].unsqueeze(1) + text_idx
        text_flat_indices = torch.where(text_mask, text_flat_indices, torch.tensor(-1, device=device, dtype=dtype))

        # --------------------------
        # 3. 向量化提取数据（修复无效数据干扰）
        # --------------------------
        # 提取有效点云数据并填充到批次张量
        valid_point_indices = point_flat_indices.masked_select(point_mask)  # [total_valid_points]
        pc_valid = point_cloud[valid_point_indices]  # [total_valid_points, 3]
        pc_batch = torch.zeros(batch_size, max_points, 3, device=device, dtype=point_cloud.dtype)
        pc_batch.masked_scatter_(point_mask.unsqueeze(-1), pc_valid.unsqueeze(1))  # 按掩码填充有效数据

        # 提取有效点云特征
        pc_feat_valid = point_features[valid_point_indices]  # [total_valid_points, feat_dim]
        pc_feat_batch = torch.zeros(batch_size, max_points, point_features.size(1), 
                                   device=device, dtype=point_features.dtype)
        pc_feat_batch.masked_scatter_(point_mask.unsqueeze(-1), pc_feat_valid.unsqueeze(1))

        # 提取有效文本特征
        valid_text_indices = text_flat_indices.masked_select(text_mask)  # [total_valid_texts]
        text_valid = text_features[valid_text_indices]  # [total_valid_texts, text_feat_dim]
        text_batch = torch.zeros(batch_size, max_texts, text_features.size(1), 
                                device=device, dtype=text_features.dtype)
        text_batch.masked_scatter_(text_mask.unsqueeze(-1), text_valid.unsqueeze(1))

        # --------------------------
        # 4. 特征处理与注意力计算（关键修复：移除所有掩码参数）
        # --------------------------
        # 点云特征嵌入与投影
        pc_embedded = self.pc_embedder(pc_batch)  # [B, M, embed_dim]
        pc_features = torch.cat([pc_embedded, pc_feat_batch], dim=-1)  # [B, M, embed_dim+feat_dim]
        pc_features = self.pc_feature_proj(pc_features)  # [B, M, width]
        # 对无效点云特征置零（核心：替代注意力掩码）
        pc_features = pc_features * point_mask.unsqueeze(-1)  # [B, M, width]

        # 文本特征投影
        text_features_proj = self.text_feature_proj(text_batch)  # [B, T, width]
        # 对无效文本特征置零（核心：替代注意力掩码）
        text_features_proj = text_features_proj * text_mask.unsqueeze(-1)  # [B, T, width]

        # 第一个Cross Attention：仅传递2个参数（query和context）
        text_part_features = self.first_cross_attn(text_features_proj, pc_features)  # [B, T, width]
        # 再次对结果置零（确保无效位置不传播）
        text_part_features = text_part_features * text_mask.unsqueeze(-1)

        # Self Attention：文本内部（对特征置零）
        refined_features = self.self_attn(text_part_features)  # [B, T, width]
        refined_features = refined_features * text_mask.unsqueeze(-1)  # 再次置零

        # 第二个Cross Attention：仅传递2个参数
        final_features = self.second_cross_attn(pc_features, refined_features)  # [B, M, width]
        # 再次对结果置零
        final_features = final_features * point_mask.unsqueeze(-1)

        # --------------------------
        # 5. 输出还原
        # --------------------------
        final_features = self.ln_post(final_features)
        batch_output = self.output_proj(final_features)  # [B, M, output_dim]

        # 提取有效输出并还原为原始格式 [total_points, output_dim]
        valid_output = batch_output.masked_select(point_mask.unsqueeze(-1)).reshape(-1, self.cfg.output_dim)
        outputs = torch.zeros(point_cloud.size(0), self.cfg.output_dim, device=device, dtype=batch_output.dtype)
        outputs[valid_point_indices] = valid_output  # 直接填充有效结果

        # 验证输出维度一致性
        assert outputs.size(0) == point_cloud.size(0), \
            f"输出长度 {outputs.size(0)} 与输入点云数量 {point_cloud.size(0)} 不匹配"

        return outputs
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> torch.FloatTensor:
        if self.cfg.use_checkpoint and self.training:
            return checkpoint(
                self._forward, 
                (point_cloud, point_features, text_features, offset, mask_offset), 
                self.parameters(), 
                self.cfg.use_checkpoint
            )
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)'''
    


'''############ 这个是batch * data size  ， feats size 组织得，通过参数offset 和mask_offset 记录了每个batch结束得index ；
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from einops import repeat, rearrange

from release_module.decoder.transformers.perceiver_1d import Perceiver
from release_module.decoder.transformers.attention import ResidualCrossAttentionBlock
from release_module.decoder.utils.checkpoint import checkpoint
from release_module.decoder.utils.base import BaseModule

from release_module.decoder.autoencoders.michelangelo_autoencoder import get_embedder


@dataclass
class PointCloudTextTransformerConfig(BaseModule.Config):
    # 点云相关配置
    point_cloud_embed_type: str = "fourier"
    point_cloud_num_freqs: int = 8 # 6
    point_cloud_include_pi: bool = False # True
    
    # 模型维度配置
    feature_dim: int = 768  # 点云和文本的特征维度
    width: int = 768        # 模型隐藏层维度
    heads: int = 12         # 注意力头数
    num_self_attn_layers: int = 16  # 6  # Self Attention层数   ????????
    
    # 注意力配置
    init_scale: float = 0.25
    qkv_bias: bool = False # True
    use_flash: bool = True # False
    use_checkpoint: bool = False
    
    # 输出配置
    output_dim: int = 1  # 预测输出维度


class PointCloudTextTransformer(BaseModule):
    Config = PointCloudTextTransformerConfig
    
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
        
        # 输出投影层和LayerNorm
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        self.ln_post = nn.LayerNorm(self.cfg.width)
    

    def _forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> torch.FloatTensor:
        # 确定批次数量
        batch_size = offset.size(0)
        
        # 处理每个批次的数据
        outputs = torch.zeros(point_cloud.size(0), self.cfg.output_dim, device=point_cloud.device)
        prev_offset = 0
        prev_mask_offset = 0
        
        for i in range(batch_size):
            # 获取当前批次的点云数据范围
            curr_offset = offset[i]
            curr_mask_offset = mask_offset[i]
            
            # 分割当前批次的点云数据
            pc_batch = point_cloud[prev_offset:curr_offset]
            pc_feat_batch = point_features[prev_offset:curr_offset]
            
            # 分割当前批次的文本特征
            text_batch = text_features[prev_mask_offset:curr_mask_offset]

            # 对所有得数据进行升维度
            pc_batch = pc_batch.unsqueeze(0)
            pc_feat_batch = pc_feat_batch.unsqueeze(0)
            text_batch = text_batch.unsqueeze(0)
 
            
            # 处理点云特征
            pc_embedded = self.pc_embedder(pc_batch)
            pc_features = torch.cat([pc_embedded, pc_feat_batch], dim=-1)
            pc_features = self.pc_feature_proj(pc_features)
            
            # 处理文本特征
            text_features_proj = self.text_feature_proj(text_batch)
            
            # 第一个Cross Attention
            text_part_features = self.first_cross_attn(
                text_features_proj,  # 作为query的张量
                pc_features          # 作为context的张量
            )
            
            # Self Attention
            refined_features = self.self_attn(text_part_features)
            
            # 第二个Cross Attention
            final_features = self.second_cross_attn(
                pc_features,          # 作为query的张量
                refined_features      # 作为context的张量
            )
            
            # 输出预测
            final_features = self.ln_post(final_features)
            batch_output = self.output_proj(final_features)

            # print('------------------------')
            # print('pc_batch:', pc_batch.shape)
            # print('text_features_proj:', text_features_proj.shape)
            # print('pc_features:', pc_features.shape)
            # print('text_part_features:', text_part_features.shape)
            # print('refined_features:', refined_features.shape)
            # print('final_features:', final_features.shape)
            # print('batch_output:', batch_output.shape)
            
            outputs[prev_offset:curr_offset] = batch_output

            # 更新偏移量
            prev_offset = curr_offset
            prev_mask_offset = curr_mask_offset
        
        # 验证输出长度是否与输入点云数量匹配，确保可以用相同的offset切割
        assert outputs.size(0) == point_cloud.size(0), \
            f"输出长度 {outputs.size(0)} 与输入点云数量 {point_cloud.size(0)} 不匹配，无法用offset正确切割"
        
        return outputs
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor,
                offset: torch.LongTensor,
                mask_offset: torch.LongTensor) -> torch.FloatTensor:
        if self.cfg.use_checkpoint and self.training:
            return checkpoint(
                self._forward, 
                (point_cloud, point_features, text_features, offset, mask_offset), 
                self.parameters(), 
                self.cfg.use_checkpoint
            )
        else:
            return self._forward(point_cloud, point_features, text_features, offset, mask_offset)'''






'''
############ 下面内容是按 batch， pts size ， feat size 作为输入得
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from einops import repeat, rearrange

from release_module.decoder.transformers.perceiver_1d import Perceiver
from release_module.decoder.transformers.attention import ResidualCrossAttentionBlock
from release_module.decoder.utils.checkpoint import checkpoint
from release_module.decoder.utils.base import BaseModule

from release_module.decoder.autoencoders.michelangelo_autoencoder import get_embedder


@dataclass
class PointCloudTextTransformerConfig(BaseModule.Config):
    # 点云相关配置
    point_cloud_embed_type: str = "fourier"
    point_cloud_num_freqs: int = 8 # 6
    point_cloud_include_pi: bool = False # True
    
    # 模型维度配置
    feature_dim: int = 768  # 点云和文本的特征维度
    width: int = 768        # 模型隐藏层维度
    heads: int = 12         # 注意力头数
    num_self_attn_layers: int = 16  # 6  # Self Attention层数   ????????
    
    # 注意力配置
    init_scale: float = 0.25
    qkv_bias: bool = False # True
    use_flash: bool = True # False
    use_checkpoint: bool = False
    
    # 输出配置
    output_dim: int = 1  # 预测输出维度


class PointCloudTextTransformer(BaseModule):
    Config = PointCloudTextTransformerConfig
    
    def __init__(self, cfg=None):
        # # 打印初始化时的配置信息（移除DictConfig相关判断）
        # print("\n===== 模型初始化 =====")
        # print(f"传入的cfg类型: {type(cfg)}")
        # print(f"是否为映射类型: {isinstance(cfg, dict)}")  # 仅判断是否为字典
        # if cfg is not None:
        #     if hasattr(cfg, '__class__'):
        #         print(f"cfg的类名: {cfg.__class__.__name__}")
        #     else:
        #         print(f"cfg的类名: 基础字典类型")
        #     # 打印配置中的键（而非属性，兼容字典）
        #     if isinstance(cfg, dict):
        #         print(f"cfg的键: {list(cfg.keys())[:5]}...")  # 只打印前5个键
        #     else:
        #         print(f"cfg的属性: {dir(cfg)[:5]}...")  # 对于dataclass实例打印属性
            
        super().__init__(cfg)
    
    def configure(self) -> None:
        super().configure()
        
        # # 打印配置解析后的信息
        # print("\n===== 配置解析后 =====")
        # print(f"self.cfg类型: {type(self.cfg)}")
        # print(f"self.cfg的类名: {self.cfg.__class__.__name__}")
        # print(f"特征维度配置: {self.cfg.feature_dim}")
        # print(f"隐藏层维度配置: {self.cfg.width}")
        # print(f"嵌入类型: {self.cfg.point_cloud_embed_type}")
        
        # 初始化点云坐标嵌入器
        self.pc_embedder = get_embedder(
            embed_type=self.cfg.point_cloud_embed_type,
            num_freqs=self.cfg.point_cloud_num_freqs,
            input_dim=3,
            include_pi=self.cfg.point_cloud_include_pi
        )
        
        # # 打印嵌入器信息以调试
        # print(f"嵌入器类型: {type(self.pc_embedder)}")
        # print(f"嵌入器是否有out_dim属性: {hasattr(self.pc_embedder, 'out_dim')}")
        # if hasattr(self.pc_embedder, 'out_dim'):
        #     print(f"嵌入器输出维度: {self.pc_embedder.out_dim}")
        
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
        
        # 输出投影层和LayerNorm
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        self.ln_post = nn.LayerNorm(self.cfg.width)
    
    def _forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, n_points, _ = point_cloud.shape
        _, n_texts, _ = text_features.shape
        
        # 处理点云特征
        pc_embedded = self.pc_embedder(point_cloud)
        pc_features = torch.cat([pc_embedded, point_features], dim=-1)
        pc_features = self.pc_feature_proj(pc_features)
        
        # 处理文本特征
        text_features_proj = self.text_feature_proj(text_features)
        
        # 第一个Cross Attention
        text_part_features = self.first_cross_attn(
            text_features_proj,  # 作为query的张量
            pc_features          # 作为context的张量
        )
        
        # Self Attention
        refined_features = self.self_attn(text_part_features)
        
        # 第二个Cross Attention
        final_features = self.second_cross_attn(
            pc_features,          # 作为query的张量
            refined_features      # 作为context的张量
        )
        
        # 输出预测
        final_features = self.ln_post(final_features)
        outputs = self.output_proj(final_features)
        
        return outputs
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor) -> torch.FloatTensor:
        if self.cfg.use_checkpoint and self.training:
            return checkpoint(
                self._forward, 
                (point_cloud, point_features, text_features), 
                self.parameters(), 
                self.cfg.use_checkpoint
            )
        else:
            return self._forward(point_cloud, point_features, text_features)'''

'''from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from einops import repeat, rearrange

from release_module.transformers.perceiver_1d import Perceiver
from release_module.transformers.attention import ResidualCrossAttentionBlock
from release_module.utils.checkpoint import checkpoint
from release_module.utils.base import BaseModule
# from craftsman.utils.typing import *

from release_module.autoencoders.michelangelo_autoencoder import get_embedder

@dataclass
class PointCloudTextTransformerConfig(BaseModule.Config):
    # 点云相关配置
    point_cloud_embed_type: str = "fourier"
    point_cloud_num_freqs: int = 6
    point_cloud_include_pi: bool = True
    
    # 模型维度配置
    feature_dim: int = 768  # 点云和文本的特征维度
    width: int = 768        # 模型隐藏层维度
    heads: int = 12         # 注意力头数
    num_self_attn_layers: int = 6  # Self Attention层数
    
    # 注意力配置
    init_scale: float = 0.25
    qkv_bias: bool = True
    use_flash: bool = False
    use_checkpoint: bool = False
    
    # 输出配置
    output_dim: int = 1  # 预测输出维度

class PointCloudTextTransformer(BaseModule):
    Config = PointCloudTextTransformerConfig
    
    def __init__(self, cfg=None):
        # 打印初始化时的配置信息
        print("\n===== 模型初始化 =====")
        print(f"传入的cfg类型: {type(cfg)}")
        print(f"是否为映射类型: {isinstance(cfg, (dict, DictConfig))}")
        if cfg is not None:
            print(f"cfg的类名: {cfg.__class__.__name__}")
            print(f"cfg的属性: {dir(cfg)[:5]}...")  # 只打印前5个属性
            
        super().__init__(cfg)
    
    def configure(self) -> None:
        super().configure()
        
        # 打印配置解析后的信息
        print("\n===== 配置解析后 =====")
        print(f"self.cfg类型: {type(self.cfg)}")
        print(f"self.cfg的类名: {self.cfg.__class__.__name__}")
        print(f"特征维度配置: {self.cfg.feature_dim}")
        print(f"隐藏层维度配置: {self.cfg.width}")
        print(f"嵌入类型: {self.cfg.point_cloud_embed_type}")
        
        # 初始化点云坐标嵌入器
        self.pc_embedder = get_embedder(
            embed_type=self.cfg.point_cloud_embed_type,
            num_freqs=self.cfg.point_cloud_num_freqs,
            input_dim=3,
            include_pi=self.cfg.point_cloud_include_pi
        )
        
        # 打印嵌入器信息以调试
        print(f"嵌入器类型: {type(self.pc_embedder)}")
        print(f"嵌入器是否有out_dim属性: {hasattr(self.pc_embedder, 'out_dim')}")
        if hasattr(self.pc_embedder, 'out_dim'):
            print(f"嵌入器输出维度: {self.pc_embedder.out_dim}")
        
        # 点云特征投影层
        if hasattr(self.pc_embedder, 'out_dim'):
            embed_dim = self.pc_embedder.out_dim
        else:
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
        
        # 输出投影层和LayerNorm
        self.output_proj = nn.Linear(self.cfg.width, self.cfg.output_dim)
        self.ln_post = nn.LayerNorm(self.cfg.width)
    
    def _forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, n_points, _ = point_cloud.shape
        _, n_texts, _ = text_features.shape
        
        # 处理点云特征
        pc_embedded = self.pc_embedder(point_cloud)
        pc_features = torch.cat([pc_embedded, point_features], dim=-1)
        pc_features = self.pc_feature_proj(pc_features)
        
        # 处理文本特征
        text_features_proj = self.text_feature_proj(text_features)
        
        # 第一个Cross Attention - 进一步修正参数传递方式
        # 移除所有关键字参数，仅使用位置参数
        text_part_features = self.first_cross_attn(
            text_features_proj,  # 作为query的张量
            pc_features          # 作为context的张量，直接作为位置参数传递
        )
        
        # Self Attention
        refined_features = self.self_attn(text_part_features)
        
        # 第二个Cross Attention - 进一步修正参数传递方式
        final_features = self.second_cross_attn(
            pc_features,          # 作为query的张量
            refined_features      # 作为context的张量，直接作为位置参数传递
        )
        
        # 输出预测
        final_features = self.ln_post(final_features)
        outputs = self.output_proj(final_features)
        
        return outputs
    
    def forward(self, 
                point_cloud: torch.FloatTensor,
                point_features: torch.FloatTensor,
                text_features: torch.FloatTensor) -> torch.FloatTensor:
        if self.cfg.use_checkpoint and self.training:
            return checkpoint(
                self._forward, 
                (point_cloud, point_features, text_features), 
                self.parameters(), 
                self.cfg.use_checkpoint
            )
        else:
            return self._forward(point_cloud, point_features, text_features)'''
    