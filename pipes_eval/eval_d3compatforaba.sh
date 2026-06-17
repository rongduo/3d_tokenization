

# 用3dcompat 进行消融实验的测试
# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
# conda activate find3d3
# bash evalbenchmark/eval_d3compatforaba.sh

################## coarse dataset

######## find3d_ab1_partfieldloss
python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab1_partfieldloss/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab1_partfieldloss/' \
    --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab1_partfieldloss/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab1_partfieldloss/'  \
    --net_type 'net1'  --test_type 'feats' --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab1_partfieldloss/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab1_partfieldloss/'  \
    --net_type 'net1'  --test_type 'feats' --part_query  &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab1_partfieldloss/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab1_partfieldloss/'  \
    --net_type 'net1'  --test_type 'feats' &&

######## find3d_ab2_partfieldloss_sizeaug
python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab2_partfieldloss_sizeaug/' \
    --net_type 'net1'  --test_type 'feats' --part_query --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab2_partfieldloss_sizeaug/'  \
    --net_type 'net1'  --test_type 'feats' --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab2_partfieldloss_sizeaug/'  \
    --net_type 'net1'  --test_type 'feats' --part_query  &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab2_partfieldloss_sizeaug/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab2_partfieldloss_sizeaug/'  \
    --net_type 'net1'  --test_type 'feats'  &&


######## find3d_ab3_partfieldloss_sizeaug_decoder
python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab3_partfieldloss_sizeaug_decoder/' \
    --net_type 'net3'  --test_type 'feats' --part_query --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab3_partfieldloss_sizeaug_decoder/'  \
    --net_type 'net3'  --test_type 'feats' --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab3_partfieldloss_sizeaug_decoder/'  \
    --net_type 'net3'  --test_type 'feats' --part_query  &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab3_partfieldloss_sizeaug_decoder/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab3_partfieldloss_sizeaug_decoder/'  \
    --net_type 'net3'  --test_type 'feats'  &&


######## find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor
python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/' \
    --net_type 'net4'  --test_type 'feats' --part_query --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/'  \
    --net_type 'net4'  --test_type 'feats' --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/'  \
    --net_type 'net4'  --test_type 'feats' --part_query  &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab4_partfieldloss_sizeaug_decoder_canoncolor/'  \
    --net_type 'net4'  --test_type 'feats'  &&

######## find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign
python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/' \
    --net_type 'net5'  --test_type 'feats' --part_query --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/'  \
    --net_type 'net5'  --test_type 'feats' --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/'  \
    --net_type 'net5'  --test_type 'feats' --part_query  &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign/'  \
    --net_type 'net5'  --test_type 'feats'  &&


######## find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox
python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/' \
    --net_type 'net8'  --test_type 'feats' --part_query --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/'  \
    --net_type 'net8'  --test_type 'feats' --canonical &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/'  \
    --net_type 'net8'  --test_type 'feats' --part_query  &&

python -m evalbenchmark.d3compat.eval_benchmark --benchmark d3compat \
    --data_root /apdcephfs_cq11/share_303570626/lanejin/dataset/3dcompat200/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'results8ab/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/ckpt_200.pth'  --save_dir 'results8abatest/find3d_ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox/'  \
    --net_type 'net8'  --test_type 'feats'  


# ################## fine dataset
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
#     --net_type 'net1'  --test_type 'feats' 
