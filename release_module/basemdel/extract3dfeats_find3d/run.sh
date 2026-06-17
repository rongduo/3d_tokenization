
source /home/jl/anaconda3/bin/activate  /home/jl/anaconda3/envs/find3d


python -m release_module.basemdel.extract3dfeats_find3d.ex3dfeats --object_path '/data3/jl/mesh_primitive_fitting/data/cat0.9/models/model_normalized.obj' --checkpoint_path '/data4/jl/project/Find3D/model/checkpoints/ckpt_80.pth'   --save_path '/data4/jl/project/Find3D/results_module/extract3dfeats/feats.ply' 