



# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
# conda activate find3d3
# bash pipes_eval/eval_partnete.sh


gpuID=1,2,3,4

export https_proxy=http://192.168.1.36:3128 http_proxy=http://192.168.1.36:3128  all_proxy=http://192.168.1.36:3128

########### test : orgfind3d finetune on partnete
### find3d retrained test
CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
    --data_root 'dataset/datasets/datasets/test/partnet' \
    --checkpoint_path 'results/find3d_partnete/ckpt_25.pth'  --save_dir 'results8/finetuneon_partnet/' \
    --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
    --data_root 'dataset/datasets/datasets/test/partnet' \
    --checkpoint_path 'results/find3d_partnete/ckpt_25.pth'  --save_dir 'results8/finetuneon_partnet/' \
    --net_type 'net1'  --test_type 'feats' --canonical &&

CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
    --data_root 'dataset/datasets/datasets/test/partnet' \
    --checkpoint_path 'results/find3d_partnete/ckpt_25.pth'  --save_dir 'results8/finetuneon_partnet/' \
    --net_type 'net1'  --test_type 'feats' --part_query  &&

CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
    --data_root 'dataset/datasets/datasets/test/partnet' \
    --checkpoint_path 'results/find3d_partnete/ckpt_25.pth'  --save_dir 'results8/finetuneon_partnet/' \
    --net_type 'net1'  --test_type 'feats' 



#### find3d retrained test
# CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root 'dataset/datasets/datasets/test/partnet' \
#     --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results8/partnete_find3dretrain/' \
#     --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root 'dataset/datasets/datasets/test/partnet' \
#     --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results8/partnete_find3dretrain/' \
#     --net_type 'net1'  --test_type 'feats' --canonical &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root 'dataset/datasets/datasets/test/partnet' \
#     --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results8/partnete_find3dretrain/' \
#     --net_type 'net1'  --test_type 'feats' --part_query  &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m pipes_eval.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root 'dataset/datasets/datasets/test/partnet' \
#     --checkpoint_path 'dataset/checkpoints/find3d_3dcompat.pth'  --save_dir 'results8/partnete_find3dretrain/' \
#     --net_type 'net1'  --test_type 'feats' 

# ################

# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/partnete_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query --canonical &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/partnete_ours/' \
#     --net_type 'net8'  --test_type 'feats' --canonical &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/partnete_ours/' \
#     --net_type 'net8'  --test_type 'feats' --part_query  &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth'  --save_dir 'results8/partnete_ours/' \
#     --net_type 'net8'  --test_type 'feats' 


# ##
# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' --part_query --canonical &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' --canonical &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' --part_query  &&

# CUDA_VISIBLE_DEVICES=$gpuID python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results6/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results5/test_final/' \
#     --net_type 'net8'  --test_type 'feats' 


#################### 基于3dcompat coarse 对不同架构进行消融
# # aba1
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab1_partfieldloss/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net1'  --test_type 'feats' &&

# # aba2
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net2'  --test_type 'feats' &&

# # aba3 decoder
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net3' --test_type 'feats'   &&
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net3' --test_type 'pre'   &&

# # aba4 decoder caonocolor
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net4' --test_type 'feats'   &&
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net4' --test_type 'pre'   &&

# # aba5 decoder caonocolor catealign
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net5' --test_type 'feats'  &&
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net5' --test_type 'pre'  &&

# # aba6 decoder bbox
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net6'  --test_type 'feats'  &&
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results5/find3d_ab6_partfieldloss_sizeaug_decoder_bbox/ckpt_200.pth'  --save_dir 'results5/test/' \
#     --net_type 'net6'  --test_type 'pre'














###################### 不同训练数据集的消融
# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_all/ckpt_200.pth' \
#     --save_dir 'results_tmp/partnete/'  &&

# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_coarse/ckpt_200.pth' \
#     --save_dir 'results_tmp/partnete2/'

# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results4/find3d_wopremodel_halfd3com_fine/ckpt_200.pth' \
#     --save_dir 'results_tmp/partnete3/'


# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results4/find3d_halfd3com_all/ckpt_200.pth' \
#     --save_dir 'results_tmp/partnete4/'

# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results4/find3d_halfd3com_fine/ckpt_200.pth' \
#     --save_dir 'results_tmp/partnete5/'


# python -m evalbenchmark.partnete.eval_benchmark --benchmark PartNetE \
#     --data_root /apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/test \
#     --checkpoint_path 'results4/find3d_halfd3com_coarse/ckpt_200.pth' \
#     --save_dir 'results_tmp/partnete6/'
