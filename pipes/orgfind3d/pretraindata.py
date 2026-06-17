"""
将ruoyu 训练table的数据集进行训练  （from partnet）
1. 测试，先用训练集中的数据看能不能成功拟合
2. 找一些别的数据集，测试看效果
"""

import json
import os



useobjjson = '/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/table/passed_ids_table.json'
dataroot = '/apdcephfs_cq11/share_303570626/lanejin/dataset/partnet/forfind3dtrain'


with open(useobjjson, 'r', encoding='utf-8') as f:
    data = json.load(f)

datapaths = []
for objname in data:
    datapath = os.path.join(dataroot, str(objname))
    datapaths.append(datapath)


# 保存 datapaths 到 txt 文件
output_txt = "/apdcephfs_cq11/share_303570626/lanejin/project/Find3D/dataset/table/train.txt"  # 输出文件路径，可自定义
with open(output_txt, "w", encoding="utf-8") as f:
    for path in datapaths:
        f.write(path + "\n")  # 每行写入一个路径

print(f"已将 {len(datapaths)} 个路径保存到 {output_txt}")
