












#####多卡并行 ; 学习率的调整，线性，batchsize增加的倍数*gpu数增加的倍数*原本的学习率
# 尝试加载预训练模型  0.003
# bash pipes/orgfind3d/run_train.sh
export https_proxy=http://192.168.1.36:3128 http_proxy=http://192.168.1.36:3128  all_proxy=http://192.168.1.36:3128

gpuID=0,1,2,3,4,5,6,7
echo "Using GPU ID: $gpuID"
dataset=partnete
expername=partnete

CUDA_VISIBLE_DEVICES=$gpuID /x2robot_v2/lanejin/new_data/miniforge3/envs/find3d/bin/python -m pipes.orgfind3d.train_mutigpus \
    --data_root 'dataset/datasets/'$dataset \
    --ckpt_dir='results' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=64 \
    --n_epoch=200 \
    --exp_suffix=$expername \
    --pretrained_path 'dataset/checkpoints/orgfind3d.pth'


# gpuID=0,1,2,3,4,5
# echo "Using GPU ID: $gpuID"
# expername=d3compat

# CUDA_VISIBLE_DEVICES=$gpuID python -m pipes.orgfind3d.train_mutigpus \
#     --data_root 'dataset/'$expername \
#     --ckpt_dir='results' \
#     --lr=0.0005 \
#     --eta_min=0.00005 \
#     --batch_size=256 \
#     --n_epoch=200 \
#     --exp_suffix=$expername \
#     --pretrained_path 'dataset/checkpoints/find3d_3dcompat.pth'
    # --n_epoch=80 \
    # --batch_size=256 \
    # --pretrained_path 'model/checkpoints/ckpt_80.pth'














# 训练，不用加载预训练模型
# gpuID=6
# echo "Using GPU ID: $gpuID"
# CUDA_VISIBLE_DEVICES=$gpuID python model/training/train.py --data_root 'dataset/labeled' --ckpt_dir='results' --lr=0.0003 --eta_min=0.00005 --batch_size=64 --n_epoch=80 --exp_suffix=train_test1



# # 尝试加载预训练模型
# gpuID=0,1,2,3,4,5
# echo "Using GPU ID: $gpuID"
# CUDA_VISIBLE_DEVICES=$gpuID python -m model.training.train \
#     --data_root 'dataset/' \
#     --ckpt_dir='results' \
#     --lr=0.0003 \
#     --eta_min=0.00005 \
#     --batch_size=256 \
#     --n_epoch=80 \
#     --exp_suffix=traindata_partverse_3Dcom \
#     --pretrained_path 'model/checkpoints/ckpt_80.pth'




######多卡并行 ; 学习率的调整，线性，batchsize增加的倍数*gpu数增加的倍数*原本的学习率
# 尝试加载预训练模型  0.003
# gpuID=0,1,2,3,4,5
# echo "Using GPU ID: $gpuID"
# CUDA_VISIBLE_DEVICES=$gpuID python -m model.training.train_mutigpus \
#     --data_root 'dataset/' \
#     --ckpt_dir='results' \
#     --lr=0.002 \
#     --eta_min=0.00005 \
#     --batch_size=256 \
#     --n_epoch=80 \
#     --exp_suffix=traindata_partverse_3Dcom \
#     --pretrained_path 'model/checkpoints/ckpt_80.pth'
# ## 感觉不充分，接着训练
# gpuID=0,1,2,3,4,5
# echo "Using GPU ID: $gpuID"
# cd ..
# cd ..
# CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline.singlecate.train_mutigpus \
#     --data_root 'dataset/' \
#     --ckpt_dir='results' \
#     --lr=0.001 \
#     --eta_min=0.00005 \
#     --batch_size=256 \
#     --n_epoch=400 \
#     --exp_suffix=traindata_partverse_3Dcom_more2 \
#     --pretrained_path 'results/find3d_traindata_partverse_3Dcom/ckpt_80.pth'
#   # 如果是重头训练学习率应该是这个 ： 0.002   
#   # 如果加载模型，要修改为停止时候的学习率
#   # 0.001 'results/find3d_traindata_partverse_3Dcom/ckpt_80.pth'
