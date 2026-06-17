

#!/bin/bash

# # 开启调试模式（执行时会打印每一步命令，可选）
# set -x

# 检查输入参数数量
if [ $# -ne 1 ]; then
    echo "错误：参数数量不正确！"
    echo "用法: $0 [org|ours]"
    echo "示例: $0 org  或  $0 ours"
    exit 1
fi

TEST_MODE=$1

# 根据参数配置测试项
case ${TEST_MODE} in
    "org")
        MODULE="pipes_eval.d3compat.eval_benchmark"
        CHECKPOINT="dataset/checkpoints/find3d_3dcompat.pth"
        SAVE_DIR="results/test/"
        NET_TYPE="net1"
        ;;
    "ours")
        MODULE="pipes_eval.d3compat.eval_benchmark"
        CHECKPOINT="dataset/checkpoints/0104_color_ckpt_200.pth" #"dataset/checkpoints/ours_final.pth"
        SAVE_DIR="results/d3com_ours/"
        NET_TYPE="net8"
        ;;
    *)
        echo "错误：无效参数 '${TEST_MODE}'，请输入 org 或 ours"
        exit 1
        ;;
esac

# 通用参数
DATA_ROOT="dataset/datasets/datasets/test/3dcompat200"
D3COM_DATATYPE="coarse"
TEST_TYPE="feats"

# 打印当前配置（便于调试）
echo "===== 测试配置 ====="
echo "测试模式: ${TEST_MODE}"
echo "模块路径: ${MODULE}"
echo "权重路径: ${CHECKPOINT}"
echo "保存目录: ${SAVE_DIR}"
echo "===================="

# 执行python命令
python -m ${MODULE} --benchmark d3compat \
    --data_root ${DATA_ROOT} \
    --d3com_datatype ${D3COM_DATATYPE} \
    --checkpoint_path ${CHECKPOINT} \
    --save_dir ${SAVE_DIR} \
    --net_type ${NET_TYPE} \
    --test_type ${TEST_TYPE} \
    --part_query --canonical

# 检查命令是否执行成功
if [ $? -eq 0 ]; then
    echo "测试成功完成！结果已保存至 ${SAVE_DIR}"
else
    echo "测试执行失败！请检查命令参数或路径"
    exit 1
fi

# #### 原始find3d测试
# python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results/test/' \
#     --net_type 'net1'  --test_type 'feats' --part_query --canonical

# #### ours final 测试
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'dataset/checkpoints/ours_final.pth'  --save_dir 'results/d3com_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query --canonical
