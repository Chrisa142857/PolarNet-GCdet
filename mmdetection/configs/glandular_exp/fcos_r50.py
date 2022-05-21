_base_ = [
    '..\\fcos\\fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py',
]
model = dict(
        bbox_head=dict(
          num_classes=2,))
classes = ('nGEC','AGC')
data_root = r'E:\wei\glandular1536_20211217'
work_dir = r'work_dir\fcos_r50_fpn'
gpu_id = 2
auto_resume = True
samples_per_gpu = 8
workers_per_gpu = 4

mean=[0.485*255, 0.456*255, 0.406*255]
std=[0.229*255, 0.224*255, 0.225*255]
img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), crop_type='absolute', allow_negative_crop=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
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
data = dict(
    # samples_per_gpu=samples_per_gpu,
    # workers_per_gpu=workers_per_gpu,
    train=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\train.json',
        img_prefix=data_root + '\\images',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\val.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline),
    test=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\test&neg_0&1.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline))

        
# learning policy
lr_config = dict(step=[16, 22, 38, 70])
runner = dict(type='EpochBasedRunner', max_epochs=100)