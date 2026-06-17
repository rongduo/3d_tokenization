# bash pipes_eval/canovis_d3compat.sh

python -m pipes_eval.d3compat.canonvis --benchmark d3compat \
    --data_root dataset/datasets/datasets/train/3dcompat/forfind3dtest \
    --d3com_datatype 'coarse' \
    --checkpoint_path 'dataset/checkpoints/ours_final.pth'  --save_dir 'results8/canovis/' \
    --net_type 'net8'  --test_type 'feats' --part_query --canonical