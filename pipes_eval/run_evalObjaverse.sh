# 对论文中数据集 objaverse的测试   objaverse-general
# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
# conda activate find3d3
# bash run_evalObjaverse.sh

# 'model/checkpoints/ckpt_80.pth'
# 'results/find3d_d3compat_prosegloss/ckpt_200.pth'
# 'results4/find3d_wopremodel_distest/ckpt_200.pth'
# 'results/find3d_partverse_3dcompat/ckpt_80.pth'

python -m model.evaluation.benchmark.eval_benchmark --benchmark Objaverse \
    --data_root 'dataset/Objaverse-General-Find3D/' --objaverse_split unseen \
    --checkpoint_path 'results/find3d_partverse_3dcompat/ckpt_80.pth' --save_dir 'results_tmp/infer_objaverse/'
