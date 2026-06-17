



# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
# conda activate find3d3
# bash pipes_eval/eval_shapnetpart.sh

########## 四种setting的实验结果测试; find3d retrained 
python -m pipes_eval.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
    --data_root 'dataset/datasets/datasets/test/shapenetpart' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results8/shapenetpart_find3dretrain/' \
    --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

python -m pipes_eval.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
    --data_root 'dataset/datasets/datasets/test/shapenetpart' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results5/shapenetpart_find3dretrain/' \
    --net_type 'net1'  --test_type 'feats' --canonical &&

python -m pipes_eval.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
    --data_root 'dataset/datasets/datasets/test/shapenetpart' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results5/shapenetpart_find3dretrain/' \
    --net_type 'net1'  --test_type 'feats' --part_query  &&

python -m pipes_eval.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
    --data_root 'dataset/datasets/datasets/test/shapenetpart' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results5/shapenetpart_find3dretrain/' \
    --net_type 'net1'  --test_type 'feats' 

# ########## 四种setting的实验结果测试
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/shapenetpart_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query --canonical &&

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results5/shapenetpart_ours/' \
#     --net_type 'net8'  --test_type 'feats' --canonical &&

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results5/shapenetpart_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query  &&

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results5/shapenetpart_ours/' \
#     --net_type 'net8'  --test_type 'feats' 


# ## 
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' --part_query --canonical &&

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' --canonical &&

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' --part_query  &&

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' 



############################ 修改了逻辑，如果要跑canonical的 ，加上 --canonical
# #################### 基于3dcompat coarse 对不同架构进行消融
# # aba1
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab1_partfieldloss/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net1'  --test_type 'feats' &&

# # aba2
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net2'  --test_type 'feats' &&

# # aba3 decoder
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net3' --test_type 'feats'   &&
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net3' --test_type 'pre'   &&

# # aba4 decoder caonocolor
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net4' --test_type 'feats'   &&
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net4' --test_type 'pre'   &&

# # aba5 decoder caonocolor catealign
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net5' --test_type 'feats'  &&
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net5' --test_type 'pre'  &&

# # aba6 decoder bbox
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net6'  --test_type 'feats'  &&
# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net6'  --test_type 'pre'






###################### 不同训练数据集的消融

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_all/ckpt_200.pth' --save_dir 'results_tmp/infer_objaverse/'  &&


# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_coarse/ckpt_200.pth' --save_dir 'results_tmp/infer_objaverse/'


# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_fine/ckpt_200.pth' --save_dir 'results_tmp/infer_objaverse/'


# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_200.pth'  --save_dir 'results_tmp/infer_objaverse/'

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results4/find3d_halfd3com_fine/ckpt_200.pth'  --save_dir 'results_tmp/infer_objaverse/'

# python -m evalbenchmark.shapnetpart.eval_benchmark --benchmark ShapeNetPart \
#     --data_root '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/shapenetpart/shapenetpart_hdf5_2048' \
#     --checkpoint_path 'results4/find3d_halfd3com_coarse/ckpt_200.pth'  --save_dir 'results_tmp/infer_objaverse/'
