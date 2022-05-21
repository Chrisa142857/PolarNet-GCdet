_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
input_size = 1024
model = dict(
    neck=dict(
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        last_kernel_size=4),
    bbox_head=dict(
        num_classes=2,
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]])))
# dataset settings
classes = ('nGEC','AGC')
data_root = r'E:\wei\glandular1536_20211217'
# custom_imports = dict(imports=['glandular_trans'], allow_failed_imports=False)

work_dir = r'work_dir\ssd1024_vgg16'
gpu_id = 3
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
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
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
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\train_0&1.json',
        img_prefix=data_root + '\\images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\val_0&1.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\test&ood_0&1.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=2e-4, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
