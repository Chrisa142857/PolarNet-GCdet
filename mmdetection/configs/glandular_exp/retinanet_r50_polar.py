_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

model = dict(
    # neck=dict(
    #     type='PolarFPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=1,
    #     add_extra_convs='on_input',
    #     num_outs=5,
    #     polar_repeatnum=2),
    neck=dict(
        type='PolarNeck',
        out_channels=2048,
        polar_repeatnum=1),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=2048,
        # in_channels=256,
        stacked_convs=1,
        feat_channels=2048,
        # feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            # strides=[8, 16, 32, 64, 128]),
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),    
        ############ polar #############
        use_polar=True,
        polar_scale=1/32,
        polar_alpha=0,
        ############ polar #############
        ),)

classes = ('nGEC','AGC')
data_root = r'E:\wei\glandular1536_20211217'
# custom_imports = dict(imports=['glandular_trans'], allow_failed_imports=False)

work_dir = r'work_dir\retinanet_r50_polar'
gpu_id = 2
auto_resume = True
samples_per_gpu = 8
workers_per_gpu = 4

mean=[0.485*255, 0.456*255, 0.406*255]
std=[0.229*255, 0.224*255, 0.225*255]
img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), crop_type='absolute', allow_negative_crop=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 16, 22])
runner = dict(max_epochs=24)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\train_0&1.json',
        img_prefix=data_root + '\\images',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\val_0&1.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline),
    test=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\test&ood_0&1.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline))