_base_ = [
    '..\\centernet\\centernet_resnet18_dcnv2_140e_coco.py',
]

model = dict(
    neck=dict(use_dcn=False),
    bbox_head=dict(num_classes=2),
    test_cfg=dict(topk=10, score_thr=0.1)
    )

classes = ('nGEC','AGC')
data_root = r'E:\wei\glandular1536_20211217'
# custom_imports = dict(imports=['glandular_trans'], allow_failed_imports=False)

work_dir = r'work_dir\centernet_r50'
auto_resume = True
samples_per_gpu = 16
workers_per_gpu = 4

mean=[0.485*255, 0.456*255, 0.406*255]
std=[0.229*255, 0.224*255, 0.225*255]
img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), crop_type='absolute', allow_negative_crop=True),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'ori_filename', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[9, 18, 24])  # the real step is [9*5, 18*5, 24*5]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(dataset=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\train.json',
        img_prefix=data_root + '\\images',
        pipeline=train_pipeline)),
    val=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\val.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline),
    test=dict(
        classes=classes,
        ann_file=data_root + '\\labels\\annotations\\test_0&1.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline))