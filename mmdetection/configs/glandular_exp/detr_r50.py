_base_ = [
    # '..\\detr\\detr_r50_8x2_150e_coco.py',
    '..\\deformable_detr\\deformable_detr_twostage_refine_r50_16x2_50e_coco.py',
]

model = dict(
        bbox_head=dict(
          num_classes=2,
          loss_cls=dict(
              type='FocalLoss',
              use_sigmoid=True,
              gamma=2.0,
              alpha=0.25,
              loss_weight=2.0),
          ),
        # init_cfg=dict(type='Pretrained', checkpoint=),
        train_cfg=dict(cls_cost=dict(type='ClassificationCost', weight=1.)),
        test_cfg=dict(max_per_img=10)
        )

classes = ('nGEC','AGC')
data_root = r'E:\wei\glandular1536_20211217'
# custom_imports = dict(imports=['glandular_trans'], allow_failed_imports=False)

work_dir = r'work_dir\detr_r50'
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
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             # dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=1),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# optimizer = dict(lr=0.0000001)

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
        ann_file=data_root + '\\labels\\annotations\\test.json',
        img_prefix=data_root + '\\cropped_images',
        pipeline=val_pipeline))