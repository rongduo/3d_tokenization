
gpuID=0,1,2,3,4,5
# cd ..
# cd ..



# 物体 ID 列表
table_ids=(
    18817 18822 18827 18831 18844 
    18846 18875 18885
)

# # debug
# table_ids=(
#     18817
# )

# 固定参数（根据实际路径修改）
object_base_path='/apdcephfs_cq11/share_303570626/lanejin/dataset/partnet/forfind3dtrain/'
checkpoint_path='results/find3d_single_obj/ckpt_200.pth' # 'results/find3d_single_obj/ckpt_80.pth' # 'results/find3d_single_table/ckpt_80.pth' # 'results/find3d_train_test1/ckpt_80.pth'
mode='segmentation'
save_base_path='results_tmp/woRot/'

# 创建保存结果的目录（若不存在）
mkdir -p "$save_base_path"

# 遍历所有物体 ID 执行评估
for id in "${table_ids[@]}"; do
    echo "====================================="
    echo "开始处理物体 ID: $id"
    
    # 构建当前物体的路径
    object_path="${object_base_path}${id}/"
    savepath="${save_base_path}${id}_seg.ply"
    
    # 执行评估命令
    echo "Using GPU ID: $gpuID"
    CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline.singlecate.eval_visualize \
        --object_path "$object_path" \
        --checkpoint_path "$checkpoint_path" \
        --mode "$mode" \
        --savepath "$savepath"
    
    # 检查命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "物体 ID: $id 处理完成，结果保存至: $savepath"
    else
        echo "❌ 物体 ID: $id 处理失败！"
    fi
    echo "====================================="
    echo
done

echo "所有物体处理完毕！"
