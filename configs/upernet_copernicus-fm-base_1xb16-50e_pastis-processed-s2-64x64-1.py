_base_ = ['./default_runtime.py']

custom_imports = dict(
    imports=['copernicusbench'], #动态导入用户自定义的 Python 模块
    allow_failed_imports=False)
# 针对 PASTIS 数据集预处理版本的自定义 Dataset，需处理 Sentinel-2 13 波段时序影像
dataset_type = 'PASTISProcessedS2Dataset'  
data_root = '/mnt/ht2-nas2/wj/PASTIS_evel/dataset/dataset_for_OEF_128'
crop_size = (128, 128)  # (64, 64)
ignore_index = 255  # 255
patch_area = (16 * 10 / 1000)**2
copernicus_fm_checkpoint = (
    '/mnt/ht2-nas2/EO_test/cyz/Copernicus-FM/weights/'
    'CopernicusFM_ViT_base_varlang_e100.pth')

# OlmoEarth processed PASTIS saves 13 eval bands:
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12.
# Keep all bands to match the existing CopernicusBench Sentinel-2 configs.
pastis_s2_band_indices = list(range(13))
s2_band_wavelengths = [
    440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190
]
s2_band_bandwidths = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]

copernicus_s2_stats = dict(
    mean=[
        1201.6453, 1201.6453, 1398.6401, 1452.1696, 1783.1473, 2698.7830, 
        3022.3530, 3164.7271, 3270.4741, 3270.4741, 2392.7998, 2392.7998, 1632.4838
    ],
    std=[
        1254.5342, 1254.5342, 1200.8134, 1260.5355, 1188.0682, 1163.6321, 
        1220.4385, 1237.6727, 1232.5127, 1232.5127, 930.8286, 930.8286, 829.1475
    ],
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
)

train_pipeline = [
    dict(
        # 自定义PASTIS数据集加载，从文件读取13波段时序遥感影像。
        type='LoadPastisProcessedS2TimeSeriesFromFile',
        band_indices=pastis_s2_band_indices,
        patch_area=patch_area),
    # 自定义PASTIS数据集标签的读取方法
    dict(type='LoadPastisProcessedAnnotations', ignore_index=ignore_index),
    # 支持任意波段数的归一化
    dict(
        type='NormalizeMultibandImage',
        mean=copernicus_s2_stats['mean'],
        std=copernicus_s2_stats['std']),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(
        type='RandomRotate',
        prob=0.5,
        degree=90,
        pad_val=0,
        seg_pad_val=ignore_index),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]
test_pipeline = [
    dict(
        type='LoadPastisProcessedS2TimeSeriesFromFile',
        band_indices=pastis_s2_band_indices,
        patch_area=patch_area),
    dict(
        type='NormalizeMultibandImage',
        mean=copernicus_s2_stats['mean'],
        std=copernicus_s2_stats['std']),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadPastisProcessedAnnotations', ignore_index=ignore_index),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'reduce_zero_label', 'copernicus_meta')),
]
find_unused_parameters=True
model = dict(
    # 自定义的分割模型封装类，支持 Copernicus 骨干网络和时序输入
    type='TemporalCopernicusEncoderDecoder',  
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # 哥白尼骨干网络，支持多光谱波段
        type='CopernicusFMBackbone', 
        arch='base',
        frozen_exclude=[],
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=copernicus_fm_checkpoint),
        band_wavelengths=s2_band_wavelengths,
        band_bandwidths=s2_band_bandwidths,
        var_option='spectrum',
        input_mode='spectral',
        kernel_size=16,
        patch_area=patch_area,
    ),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=20,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=20,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='valid',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))
test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=65, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(by_epoch=True)
randomness = dict(seed=0)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
)
param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=1e-3,
        pct_start=0.0,
        anneal_strategy='cos',
        begin=0,
        end=100,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU',
        rule='greater',
        save_last=True))

work_dir="save/pastis_seg_lr0.0005_classes20"