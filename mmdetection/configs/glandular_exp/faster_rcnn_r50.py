_base_ = [
    '..\\_base_\\models\\faster_rcnn_r50_fpn.py',
    '..\\_base_\\datasets\\coco_detection.py',
    '..\\_base_\\schedules\\schedule_1x.py', '..\\_base_\\default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2)))

classes = ('nGEC','AGC')
data_root = r'E:\wei\glandular1536_20211217'
# custom_imports = dict(imports=['glandular_trans'], allow_failed_imports=False)

work_dir = r'work_dir\faster_rcnn_r50_fpn'
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
    step=[4, 8, 11, 16, 22])
runner = dict(max_epochs=24)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
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
        ann_file=data_root + '\\labels\\annotations\\test&ood_0&1.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline))