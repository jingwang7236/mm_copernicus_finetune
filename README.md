# 哥白尼下游任务项目

## Copernicus-FM

论文：Towards a Unified Copernicus Foundation Model for Earth Vision（ICCV 2025 Oral） 

官方 GitHub：https://github.com/zhu-xlab/Copernicus-FM 

欧洲哥白尼（Copernicus）遥感计划的通用地球视觉基础模型，能处理 Sentinel-1/2/3/5P 等多传感器数据


## pastis数据集-哥白尼-微调
export PYTHONPATH=/mnt/ht2-nas2/wj/PASTIS_evel/CEF:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="6, 7"


## pastis数据集-哥白尼-冻结训练
torchrun --nnodes=1 --nproc_per_node=2 --master_port=25419 tools/train.py configs/LP_copernicus_fm_base_frozen.py --launcher pytorch --amp

## pastis数据集-哥白尼-推理
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25416 tools/test.py configs/upernet_copernicus-fm-base_1xb16-50e_pastis-processed-s2-64x64-1.py /mnt/ht2-nas2/wj/PASTIS_evel/CEF/save/pastis_seg_lr0.0005/best_mIoU_epoch_40.pth --launcher pytorch --work-dir save/swinv2_infer



## pastis数据集-哥白尼-微调训练
torchrun --nnodes=1 --nproc_per_node=2 --master_port=25420 tools/train.py configs/LP_copernicus_fm_base_finetune.py --launcher pytorch --amp
