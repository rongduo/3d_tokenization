"""
 根据预训练模型，加载不同的网络
"""


import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import importlib  # 用于动态导入不同路径的网络类


class BasePointSegNet(ABC, nn.Module):
    def __init__(self, pretrained_path: str, net_config: dict):
        """
        点云语义分割网络抽象基类
        Args:
            pretrained_path: 预训练模型权重路径
            net_config: 网络专属配置（含导入路径、类名、模型参数）
                        示例：{"import_path": "model.backbone.pt3.model", 
                               "class_name": "PointSemSeg", 
                               "model_kwargs": {"num_classes": 10}}
        """
        super().__init__()
        self.pretrained_path = pretrained_path
        self.net_config = net_config
        self.net = None  # 存储初始化后的网络实例
        
        # 核心步骤：动态导入网络类 + 初始化网络 + 加载预训练
        self._dynamic_import_net()
        self._init_network()
        self._load_pretrained()
        self.eval()  # 测试模式（冻结BN、Dropout）

    @abstractmethod
    def _init_network(self):
        """抽象方法：子类实现网络初始化（适配不同类的参数差异）"""
        pass

    def _dynamic_import_net(self):
        """动态导入网络类：根据net_config的import_path和class_name导入"""
        try:
            # 1. 导入模块（如"model.backbone.pt3.model"）
            net_module = importlib.import_module(self.net_config["import_path"])
            # 2. 从模块中获取网络类（如"PointSemSeg"）
            self.NetClass = getattr(net_module, self.net_config["class_name"])
        except ImportError as e:
            raise ValueError(f"网络模块导入失败：{self.net_config['import_path']}，错误：{e}")
        except AttributeError as e:
            raise ValueError(f"模块中无该类：{self.net_config['class_name']}，错误：{e}")

    def _load_pretrained(self):
        """加载预训练模型：兼容单GPU/多GPU保存的权重"""
        try:
            checkpoint = torch.load(self.pretrained_path, map_location="cpu")
            # 处理多GPU训练的权重（去除"module."前缀）
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            # 加载权重（strict=False：允许部分参数不匹配，如新增的decoder层）
            self.net.load_state_dict(checkpoint, strict=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"预训练模型不存在：{self.pretrained_path}")

    def preprocess(self, point_cloud: dict) -> dict:
        """点云预处理：统一处理输入（如坐标归一化、特征标准化）"""
        # 假设输入是字典格式：{"xyz": (N, 3), "feat": (N, C)}（N为点数量，C为特征维度）
        xyz = point_cloud["xyz"].float()
        feat = point_cloud["feat"].float() if "feat" in point_cloud else None

        # 1. 坐标归一化（点云任务常用：将点云中心移至原点，缩放至单位球）
        centroid = torch.mean(xyz, dim=0, keepdim=True)
        xyz = xyz - centroid
        max_dist = torch.max(torch.norm(xyz, dim=1))
        xyz = xyz / max_dist  # 缩放至单位球内

        # 2. 特征标准化（若有特征）
        if feat is not None:
            feat = (feat - torch.mean(feat, dim=0)) / (torch.std(feat, dim=0) + 1e-6)

        return {"xyz": xyz.unsqueeze(0), "feat": feat.unsqueeze(0) if feat is not None else None}
        # 补batch维度（测试时单样本输入：(1, N, 3)和(1, N, C)）

    def postprocess(self, seg_logits: torch.Tensor) -> torch.Tensor:
        """后处理：输出语义分割结果（softmax求概率，取预测类别）"""
        # seg_logits shape: (1, N, num_classes)（batch=1，N为点数量）
        seg_probs = torch.softmax(seg_logits, dim=2)  # 每个点的类别概率
        seg_preds = torch.argmax(seg_probs, dim=2)    # 每个点的预测类别（0~num_classes-1）
        return seg_preds.squeeze(0)  # 去除batch维度：(N,)

    def forward(self, point_cloud: dict) -> torch.Tensor:
        """统一前向传播：适配点云网络的输入格式（xyz+feat）"""
        return self.net(point_cloud)

    def predict(self, point_cloud: dict) -> torch.Tensor:
        """统一预测接口：外部调用仅需传入点云，自动完成全流程"""
        with torch.no_grad():  # 测试时禁用梯度，加速计算
            processed_pc = self.preprocess(point_cloud)
            seg_logits = self.forward(processed_pc)
            seg_preds = self.postprocess(seg_logits)
        return seg_preds
