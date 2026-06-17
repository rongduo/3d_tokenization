

# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
# conda activate find3d3
# bash pipes_eval/eval_d3compat.sh


###  find3d retrain 200 在 3dcompat上测试
# python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
#     --net_type 'net1'  --test_type 'feats' --part_query --canonical

# coarse

# ########  find3d retrain 200 在 3dcompat上测试
python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' --canonical &&

python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' --part_query  &&

python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' && 


# fine
python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'fine' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'fine' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' --canonical &&

python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'fine' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' --part_query  &&

python -m pipes_eval.d3compat.eval_benchmark --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'fine' \
    --checkpoint_path 'dataset/checkpoints/find3d_3dcompat_200.pth'  --save_dir 'results8/find3d200_d3com/' \
    --net_type 'net1'  --test_type 'feats' 


##################################################################################
################## coarse dataset

# ######## 原始find3d的测试
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_coarse_org/' \
#     --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_coarse_org/' \
#     --net_type 'net1'  --test_type 'feats' --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_coarse_org/' \
#     --net_type 'net1'  --test_type 'feats' --part_query  &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_coarse_org/' \
#     --net_type 'net1'  --test_type 'feats' && 

# ######## ours
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_coarse_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_coarse_ours/' \
#     --net_type 'net8'  --test_type 'feats' --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_coarse_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query  &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'coarse' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_coarse_ours/' \
#     --net_type 'net8'  --test_type 'feats'  &&

# ################## fine dataset
# ######## 原始find3d的测试
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_fine_org/' \
#     --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_fine_org/' \
#     --net_type 'net1'  --test_type 'feats' --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_fine_org/' \
#     --net_type 'net1'  --test_type 'feats' --part_query  &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_80.pth'  --save_dir 'results8/d3com_fine_org/' \
#     --net_type 'net1'  --test_type 'feats' && 

# ######## ours
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_fine_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_fine_ours/' \
#     --net_type 'net8'  --test_type 'feats' --canonical &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_fine_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query  &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --d3com_datatype 'fine' \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/d3com_fine_ours/' \
#     --net_type 'net8'  --test_type 'feats'  




############### 下面如果要跑canonical的结果，要加上 --canonical
# ##################### 不同网络的消融
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab1_partfieldloss/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net3' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net4' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net4' \
#     --test_type 'pre' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net5' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net5' \
#     --test_type 'pre' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net6' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net6' \
#     --test_type 'pre' \
#     --save_dir 'results5/test_3dcom/'  && 

# ########### 在fine上测试
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab1_partfieldloss/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net3' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net4' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net4' \
#     --test_type 'pre' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net5' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net5' \
#     --test_type 'pre' \
#     --save_dir 'results5/test_3dcom/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net6' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_3dcom/'  && 
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net6' \
#     --test_type 'pre' \
#     --save_dir 'results5/test_3dcom/'  

# ##################### 不同训练数据集的消融
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_all/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_coarse/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&

# python -m evalbenchmark.d3compat.eval_benchmark c \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_fine/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&


# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_halfd3com_fine/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&


# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_halfd3com_coarse/ckpt_200.pth' \
#     --d3com_datatype 'coarse' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&

# ####
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_all/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  && 

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_coarse/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_fine/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&


# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&

# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_halfd3com_fine/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  &&


# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path 'results4/find3d_halfd3com_coarse/ckpt_200.pth' \
#     --d3com_datatype 'fine' \
#     --net_type 'net1' \
#     --test_type 'feats' \
#     --save_dir 'results5/test_train/'  









# # 对特征分割的结果进行测试
# expername=halfd3com_worot_aligncates_decoder_bbox
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path "results3/find3d_wopremodel_${expername}/ckpt_200.pth" \
#     --save_dir 'results_tmp/d3comcoarse/'   \
#     --d3com_datatype 'coarse' 


# # 对特征分割的结果进行测试
# expername=halfd3com_worot_aligncates_decoder_bbox
# python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
#     --checkpoint_path "results3/find3d_wopremodel_${expername}/ckpt_200.pth" \
#     --save_dir 'results_tmp/d3comcoarse/'   \
#     --d3com_datatype 'fine' 
