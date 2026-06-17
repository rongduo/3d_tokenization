"""
打印最终版本的网络信息用于写执行细节

python -m evalbenchmark.d3compat.netinfo \
    --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth' \
    --net_type net8 \
    --test_type feats
"""
import torch
import argparse
from release_pipeline6.ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox.mixdecodernet import PointSemSegWithDecoder


def print_net8_architecture_and_training_details(model, args):
    """
    打印net8网络架构信息与训练细节
    Args:
        model: 初始化后的net8模型实例
        args: 命令行参数（包含网络配置相关信息）
    """
    print("=" * 80)
    print("                      net8 网络架构与训练细节                      ")
    print("=" * 80)

    # 1. 网络架构核心信息
    print("\n【一、网络架构核心信息】")
    print(f"1.1 网络类型: {args.net_type} (PointSemSegWithDecoder)")
    print(f"1.2 特征场维度: 448-dimensional")
    print(f"1.3 Triplane（三平面）配置:")
    print(f"    - 原始空间分辨率: 512×512")
    print(f"    - 通道数: 128 channels")
    print(f"    - Transformer输入处理: 先下采样至128×128分辨率，每个像素作为一个token")
    print(f"1.4 Transformer配置:")
    print(f"    - 层数: 6 layers")
    print(f"    - 输出处理: Transformer后将特征还原为512×512分辨率的triplane")
    print(f"1.5 模型设备: {'GPU (CUDA)' if next(model.parameters()).is_cuda else 'CPU'}")

    # 2. 打印模型完整结构（PyTorch原生打印，显示层级与参数形状）
    print("\n【二、net8模型完整结构】")
    print(model)

    # 3. 训练细节（按指定需求补充）
    print("\n【三、训练细节】")
    print(f"3.1 训练设备: 8 × A100 GPUs")
    print(f"3.2 训练时长: 2 weeks")
    print(f"3.3 批次大小: 2 samples per GPU (总批次大小: {2 * 8} samples)")
    print(f"3.4 权重加载方式: {'仅加载backbone权重' if args.test_type == 'feats' else '加载完整模型权重'}")
    if args.test_type == 'feats':
        print(f"    - Backbone权重处理: 移除参数名前缀'backbone.'后加载，确保与模型结构匹配")
    print(f"3.5 模型状态: {'评估模式(eval)' if model.training is False else '训练模式(train)'}")

    # 4. 关键组件参数统计（可选：统计参数数量，增强信息完整性）
    print("\n【四、关键组件参数统计】")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"4.1 总参数数量: {total_params:,}")
    print(f"4.2 可训练参数数量: {trainable_params:,}")
    
    # 统计backbone参数（若存在backbone组件）
    if hasattr(model, 'backbone'):
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"4.3 Backbone参数数量: {backbone_params:,}")
    print("=" * 80)


if __name__ == '__main__':
    # 1. 解析命令行参数（与参考代码保持一致的参数结构）
    parser = argparse.ArgumentParser(description="net8 Network Architecture Printer")
    # 核心参数：需与net8加载逻辑匹配
    parser.add_argument("--checkpoint_path", required=True, type=str, help="net8模型权重文件路径")
    parser.add_argument("--net_type", default="net8", type=str, help="网络类型（固定为net8）")
    parser.add_argument("--test_type", default="pre", type=str, help="测试类型（feats/pre，影响权重加载方式）")
    # 其他可选参数（根据net8初始化需求补充，若模型无额外需求可设默认值）
    parser.add_argument("--d3com_datatype", default="coarse", type=str, help="D3CoMPaT数据集类型（coarse/fine）")
    parser.add_argument("--textembeds", default="clip", type=str, help="文本嵌入类型")
    args = parser.parse_args()

    # 2. 初始化net8模型（与参考代码加载逻辑完全一致）
    print(f"正在初始化net8模型...")
    model = PointSemSegWithDecoder(args=args)

    # 3. 加载模型权重（根据test_type选择加载方式，与参考代码逻辑一致）
    print(f"正在加载权重: {args.checkpoint_path}")
    pretrained_checkpoint = torch.load(args.checkpoint_path)
    pretrained_state_dict = pretrained_checkpoint["model_state_dict"]

    if args.test_type == 'feats' and args.net_type == 'net8':
        # 仅加载backbone权重（移除前缀'backbone.'）
        backbone_weights = {
            k.replace('backbone.', ''): v  
            for k, v in pretrained_state_dict.items()
            if k.startswith('backbone.')
        }
        model.backbone.load_state_dict(backbone_weights, strict=True)
    else:
        # 加载完整模型权重
        model.load_state_dict(pretrained_state_dict, strict=True)

    # 4. 设置模型为评估模式并移动到GPU（与参考代码一致）
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
    print(f"模型初始化完成（设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}）")

    # 5. 打印net8架构与训练细节
    print_net8_architecture_and_training_details(model, args)
