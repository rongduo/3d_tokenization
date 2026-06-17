
# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
#  bash release_pipeline5ab/run_muti1.sh
gpuID=2,3,4

######## 使用coarse 3dcompat 数据集 和 orgfind3d premodel训练消融实验


# 4. partfield loss + size aug + decoder + canoncolor
echo "Using GPU ID: $gpuID"
dataset=d3compat_coarse
expername=ab4_partfieldloss_sizeaug_decoder_canoncolor

CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline5ab.$expername.train_mutigpus \
    --data_root 'dataset/'$dataset \
    --ckpt_dir='results5' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=64 \
    --n_epoch=200 \
    --exp_suffix=$expername \
    --pretrained_path 'model/checkpoints/ckpt_80.pth'   &&

# 5. partfield loss + size aug + decoder + canoncolor + catesalign
echo "Using GPU ID: $gpuID"
dataset=d3compat_coarse
expername=ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign

CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline5ab.$expername.train_mutigpus \
    --data_root 'dataset/'$dataset \
    --ckpt_dir='results5' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=64 \
    --n_epoch=200 \
    --exp_suffix=$expername \
    --pretrained_path 'model/checkpoints/ckpt_80.pth'  
