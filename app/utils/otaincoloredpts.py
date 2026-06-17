"""
python -m tmp.otaincoloredpts
"""

import torch
import numpy as np
import os
from pathlib import Path

def process_single_directory(input_dir, output_root_dir):
    """
    处理单个目录的点云数据，生成带颜色的PLY文件
    :param input_dir: 输入目录（包含points.pt和rgb.pt）
    :param output_root_dir: 输出根目录
    """
    # 提取目录名作为ID（如coarse_b29_0ca、fine_b29_0cb）
    dir_name = os.path.basename(input_dir)
    # 清理可能的特殊字符（如单引号）
    # dir_id = dir_name.replace("'", "").strip()
    dir_id = dir_name.replace("'", "").strip()
    print(dir_name, dir_id)
    # asdf
    output_filename = f"coloredpts_{dir_id}.ply"
    output_path = os.path.join(output_root_dir, output_filename)

    # 定义输入文件路径
    points_path = os.path.join(input_dir, "points.pt")
    rgb_path = os.path.join(input_dir, "rgb.pt")

    try:
        # 1. 检查输入文件是否存在
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"points.pt不存在: {points_path}")
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"rgb.pt不存在: {rgb_path}")

        # 2. 读取数据并转换为numpy（强制CPU）
        points = torch.load(points_path, map_location=torch.device('cpu')).numpy()
        rgb = torch.load(rgb_path, map_location=torch.device('cpu')).numpy()

        # 3. 数据格式校验
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"点云维度错误，应为(N,3)，实际: {points.shape}")
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"颜色维度错误，应为(N,3)，实际: {rgb.shape}")
        if points.shape[0] != rgb.shape[0]:
            raise ValueError(f"点云数量({points.shape[0]})与颜色数量({rgb.shape[0]})不匹配")

        # 4. 颜色数据标准化（转为0-255 uint8）
        if np.max(rgb) <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        elif rgb.dtype in [np.float32, np.float64]:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb.astype(np.uint8), 0, 255)

        # 5. 写入PLY文件（ASCII格式，兼容性最好）
        with open(output_path, 'w', encoding='utf-8') as f:
            # PLY文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # 批量写入数据（比逐行更快）
            for pt, color in zip(points, rgb):
                x, y, z = pt
                r, g, b = color
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

        print(f"✅ 成功处理: {input_dir} -> {output_path} (点数: {points.shape[0]})")
        return True

    except FileNotFoundError as e:
        print(f"❌ 跳过 {input_dir}: 文件不存在 - {e}")
        return False
    except ValueError as e:
        print(f"❌ 跳过 {input_dir}: 数据格式错误 - {e}")
        return False
    except Exception as e:
        print(f"❌ 跳过 {input_dir}: 未知错误 - {str(e)}")
        return False

def batch_process_directories():
    # ===================== 配置参数 =====================
    # 输入目录列表（清理原路径中的单引号）
    input_dirs = [
        "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest/vase/coarse_b'29_008'",
        "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest/vase/fine_b'29_008'",
        "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest/vase/coarse_b'29_06f'",
        "/apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest/vase/fine_b'29_06f'"
    ]
    # 输出根目录
    output_root_dir = "/apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/results/colorcheck"

    # ===================== 执行批量处理 =====================
    # 创建输出目录（如果不存在）
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_root_dir}")
    print(f"📊 待处理目录数量: {len(input_dirs)}")
    print("="*80)

    # 遍历处理每个目录
    success_count = 0
    for input_dir in input_dirs:
        if process_single_directory(input_dir, output_root_dir):
            success_count += 1

    # 输出汇总信息
    print("="*80)
    print(f"📋 处理完成: 成功 {success_count}/{len(input_dirs)} 个目录")
    if success_count < len(input_dirs):
        print("⚠️  部分目录处理失败，请检查上述错误信息")
    else:
        print("🎉 所有目录处理成功！")

if __name__ == "__main__":
    # 检查依赖
    try:
        import torch
        import numpy
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请执行: pip install torch numpy")
        exit(1)
    
    # 执行批量处理
    batch_process_directories()

"""
########## 单个物体的
import torch
import numpy as np
import os

def save_colored_points_to_ply():
    # 定义基础路径
    base_dir = "/apdcephfs_cq11/share_303570626/lanejin/project/find3d_release/results/data/coarse_b29_0cb"
    
    # 定义各文件路径
    points_path = os.path.join(base_dir, "points.pt")
    rgb_path = os.path.join(base_dir, "rgb.pt")
    output_path = os.path.join(base_dir, "coloredpts.ply")

    try:
        # 1. 读取点云坐标和颜色数据
        print("正在读取点云数据...")
        # 读取并转换为numpy数组（处理可能的CUDA tensor）
        points = torch.load(points_path, map_location=torch.device('cpu')).numpy()
        rgb = torch.load(rgb_path, map_location=torch.device('cpu')).numpy()

        # 2. 数据格式校验
        print("正在校验数据格式...")
        # 检查点云维度 (N, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"点云坐标维度错误，应为(N,3)，实际为{points.shape}")
        # 检查颜色维度 (N, 3)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"颜色数据维度错误，应为(N,3)，实际为{rgb.shape}")
        # 检查点云数量和颜色数量匹配
        if points.shape[0] != rgb.shape[0]:
            raise ValueError(f"点云数量({points.shape[0]})与颜色数量({rgb.shape[0]})不匹配")

        # 3. 处理颜色数据（确保是0-255的整数）
        # 如果是0-1的浮点数，转换为0-255
        if np.max(rgb) <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        # 如果是浮点数且范围超过1，直接转整数（确保在0-255范围内）
        elif rgb.dtype in [np.float32, np.float64]:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # 确保最终是uint8类型
        else:
            rgb = rgb.astype(np.uint8)

        # 4. 写入PLY文件
        print(f"正在写入PLY文件：{output_path}")
        with open(output_path, 'w') as f:
            # PLY文件头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # 写入点云数据和颜色数据
            for i in range(points.shape[0]):
                x, y, z = points[i]
                r, g, b = rgb[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

        print(f"PLY文件保存成功！共写入{points.shape[0]}个带颜色的点云")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e.filename}")
    except ValueError as e:
        print(f"错误：数据格式问题 - {e}")
    except Exception as e:
        print(f"未知错误：{e}")
        raise

if __name__ == "__main__":
    save_colored_points_to_ply()
"""
