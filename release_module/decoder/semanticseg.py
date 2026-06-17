"""
    搭建一个decoder结构，用于 融合pts embeds 和 text embeds ； 并输出预测的part的sdf值
    python -m release_module.decoder.semanticseg
    适配find3d的版本
"""

import torch
from dataclasses import asdict
from release_module.decoder.ptstextnet import PointCloudTextTransformer, PointCloudTextTransformerConfig

def test_model_initialization():
    """测试模型是否能正确初始化（移除OmegaConf）"""
    print("\n===== 测试模型初始化 =====")
    
    # 创建原始配置（dataclass）
    cfg = PointCloudTextTransformerConfig()
    print(f"配置实例类型: {type(cfg)}")
    print(f"是否为dataclass: {hasattr(cfg, '__dataclass_fields__')}")
    
    # 转换为字典（仅使用原生dict）
    cfg_dict = asdict(cfg)
    print(f"转换为字典后的类型: {type(cfg_dict)}")
    
    # 尝试用不同方式初始化模型
    try:
        print("\n--- 尝试用原始配置（dataclass）初始化 ---")
        model = PointCloudTextTransformer(cfg)
        print("原始配置初始化成功")
    except Exception as e:
        print(f"原始配置初始化失败: {str(e)}")
        
    try:
        print("\n--- 尝试用字典配置初始化 ---")
        model = PointCloudTextTransformer(cfg_dict)
        print("字典配置初始化成功")
    except Exception as e:
        print(f"字典配置初始化失败: {str(e)}")

def test_forward_pass():
    """测试前向传播是否正常工作（移除OmegaConf，手动循环参数）"""
    # 测试参数组合（batch_size, n_points, n_texts）
    test_cases = [
        (2, 1024, 8),
        (1, 512, 4),
        (4, 2048, 16)
    ]
    
    for batch_size, n_points, n_texts in test_cases:
        print(f"\n===== 测试前向传播: {batch_size}, {n_points}, {n_texts} =====")
        
        # 用字典方式创建配置（基于dataclass转换）
        cfg = PointCloudTextTransformerConfig()
        cfg_dict = asdict(cfg)
        print(f"使用字典配置类型: {type(cfg_dict)}")
        
        # 初始化模型
        try:
            model = PointCloudTextTransformer(cfg_dict)
            model.eval()
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            continue  # 跳过当前用例
        
        # 创建输入数据
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        try:
            # 点云坐标 [B, N, 3]
            point_cloud = torch.randn(batch_size, n_points, 3, device=device)
            # 点特征 [B, N, feature_dim]
            point_features = torch.randn(batch_size, n_points, cfg_dict['feature_dim'], device=device)
            # 文本特征 [B, T, feature_dim]
            text_features = torch.randn(batch_size, n_texts, cfg_dict['feature_dim'], device=device)
        except Exception as e:
            print(f"数据创建失败: {str(e)}")
            continue
        
        print(f"点云形状: {point_cloud.shape}")
        print(f"点特征形状: {point_features.shape}")
        print(f"文本特征形状: {text_features.shape}")
        
        model = model.to(device)
        
        # 执行前向传播
        try:
            with torch.no_grad():  # 关闭梯度，加速测试
                outputs = model(point_cloud, point_features, text_features)
            print(f"输出形状: {outputs.shape}")
            
            # 检查输出形状是否符合预期 [B, N, output_dim]
            expected_shape = (batch_size, n_points, cfg_dict['output_dim'])
            if outputs.shape == expected_shape:
                print("前向传播测试成功")
            else:
                print(f"输出形状不正确，预期 {expected_shape}, 实际 {outputs.shape}")
        except Exception as e:
            print(f"前向传播测试失败: {str(e)}")

def test_checkpoint_mechanism():
    """测试检查点机制（移除OmegaConf，用dataclass+字典配置）"""
    print("\n===== 测试检查点机制 =====")
    
    # 创建带检查点配置的dataclass，再转换为字典
    try:
        cfg = PointCloudTextTransformerConfig(use_checkpoint=True)
        cfg_dict = asdict(cfg)
        print(f"检查点配置（字典）: {cfg_dict['use_checkpoint']}")  # 验证检查点是否开启
    except Exception as e:
        print(f"配置创建失败: {str(e)}")
        return
    
    # 初始化模型
    try:
        model = PointCloudTextTransformer(cfg_dict)
        model.train()  # 检查点通常在训练模式下启用
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return
    
    # 创建输入数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    batch_size, n_points, n_texts = 2, 1024, 8
    try:
        point_cloud = torch.randn(batch_size, n_points, 3, device=device)
        point_features = torch.randn(batch_size, n_points, cfg_dict['feature_dim'], device=device)
        text_features = torch.randn(batch_size, n_texts, cfg_dict['feature_dim'], device=device)
    except Exception as e:
        print(f"数据创建失败: {str(e)}")
        return
    
    print(f"点云形状: {point_cloud.shape}")
    print(f"点特征形状: {point_features.shape}")
    print(f"文本特征形状: {text_features.shape}")
    
    model = model.to(device)
    
    # 执行前向传播（训练模式，不关闭梯度）
    try:
        outputs = model(point_cloud, point_features, text_features)
        print(f"输出形状: {outputs.shape}")
        
        # 检查输出形状
        expected_shape = (batch_size, n_points, cfg_dict['output_dim'])
        if outputs.shape == expected_shape:
            print("检查点机制测试成功")
        else:
            print(f"输出形状不正确，预期 {expected_shape}, 实际 {outputs.shape}")
    except Exception as e:
        print(f"检查点机制测试失败: {str(e)}")

if __name__ == "__main__":
    cfg = PointCloudTextTransformerConfig()
    cfg_dict = asdict(cfg)
    decoder = PointCloudTextTransformer(cfg_dict)

    # # 依次执行所有测试
    # test_model_initialization()
    # test_forward_pass()
    # test_checkpoint_mechanism()
    # print("\n所有测试执行完毕")

'''import torch
import pytest
from dataclasses import asdict
from omegaconf import OmegaConf, DictConfig
from release_module.ptstextnet import PointCloudTextTransformer, PointCloudTextTransformerConfig

def test_model_initialization():
    """测试模型是否能正确初始化"""
    print("\n===== 测试模型初始化 =====")
    
    # 创建配置并打印信息
    cfg = PointCloudTextTransformerConfig()
    print(f"配置实例类型: {type(cfg)}")
    print(f"是否为dataclass: {hasattr(cfg, '__dataclass_fields__')}")
    
    # 尝试将配置转换为字典
    cfg_dict = asdict(cfg)
    print(f"转换为字典后的类型: {type(cfg_dict)}")
    
    # 尝试将配置转换为OmegaConf的DictConfig
    cfg_omega = OmegaConf.create(cfg)
    print(f"转换为OmegaConf后的类型: {type(cfg_omega)}")
    
    # 尝试用不同方式初始化模型
    try:
        print("\n--- 尝试用原始配置初始化 ---")
        model = PointCloudTextTransformer(cfg)
        print("原始配置初始化成功")
    except Exception as e:
        print(f"原始配置初始化失败: {str(e)}")
        
    try:
        print("\n--- 尝试用字典配置初始化 ---")
        model = PointCloudTextTransformer(cfg_dict)
        print("字典配置初始化成功")
    except Exception as e:
        print(f"字典配置初始化失败: {str(e)}")
        
    try:
        print("\n--- 尝试用OmegaConf初始化 ---")
        model = PointCloudTextTransformer(cfg_omega)
        print("OmegaConf初始化成功")
    except Exception as e:
        print(f"OmegaConf初始化失败: {str(e)}")

@pytest.mark.parametrize("batch_size, n_points, n_texts", [
    (2, 1024, 8),
    (1, 512, 4),
    (4, 2048, 16)
])
def test_forward_pass(batch_size, n_points, n_texts):
    """测试前向传播是否正常工作"""
    print(f"\n===== 测试前向传播: {batch_size}, {n_points}, {n_texts} =====")
    
    # 用字典方式创建配置
    cfg_dict = asdict(PointCloudTextTransformerConfig())
    print(f"使用字典配置类型: {type(cfg_dict)}")
    
    # 初始化模型
    model = PointCloudTextTransformer(cfg_dict)
    model.eval()
    
    # 创建输入数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    point_cloud = torch.randn(batch_size, n_points, 3, device=device)
    point_features = torch.randn(batch_size, n_points, cfg_dict['feature_dim'], device=device)
    text_features = torch.randn(batch_size, n_texts, cfg_dict['feature_dim'], device=device)
    
    print(f"点云形状: {point_cloud.shape}")
    print(f"点特征形状: {point_features.shape}")
    print(f"文本特征形状: {text_features.shape}")
    
    model = model.to(device)
    
    # 执行前向传播
    try:
        with torch.no_grad():
            outputs = model(point_cloud, point_features, text_features)
        print(f"输出形状: {outputs.shape}")
        
        # 检查输出形状
        assert outputs.shape == (batch_size, n_points, cfg_dict['output_dim']), \
            f"输出形状不正确，预期 {(batch_size, n_points, cfg_dict['output_dim'])}, 实际 {outputs.shape}"
        print("前向传播测试成功")
    except Exception as e:
        print(f"前向传播测试失败: {str(e)}")
        raise  # 重新抛出异常，让pytest捕获

def test_checkpoint_mechanism():
    """测试检查点机制"""
    print("\n===== 测试检查点机制 =====")
    
    # 用OmegaConf创建配置
    cfg_omega = OmegaConf.create(PointCloudTextTransformerConfig(use_checkpoint=True))
    print(f"使用OmegaConf配置类型: {type(cfg_omega)}")
    
    # 初始化模型
    model = PointCloudTextTransformer(cfg_omega)
    model.train()
    
    # 创建输入数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    batch_size, n_points, n_texts = 2, 1024, 8
    point_cloud = torch.randn(batch_size, n_points, 3, device=device)
    point_features = torch.randn(batch_size, n_points, cfg_omega.feature_dim, device=device)
    text_features = torch.randn(batch_size, n_texts, cfg_omega.feature_dim, device=device)
    
    print(f"点云形状: {point_cloud.shape}")
    print(f"点特征形状: {point_features.shape}")
    print(f"文本特征形状: {text_features.shape}")
    
    model = model.to(device)
    
    # 执行前向传播
    try:
        outputs = model(point_cloud, point_features, text_features)
        print(f"输出形状: {outputs.shape}")
        
        # 检查输出形状
        assert outputs.shape == (batch_size, n_points, cfg_omega.output_dim), \
            f"输出形状不正确，预期 {(batch_size, n_points, cfg_omega.output_dim)}, 实际 {outputs.shape}"
        print("检查点机制测试成功")
    except Exception as e:
        print(f"检查点机制测试失败: {str(e)}")
        raise  # 重新抛出异常，让pytest捕获

if __name__ == "__main__":
    pytest.main(["-v", __file__])
    '''
