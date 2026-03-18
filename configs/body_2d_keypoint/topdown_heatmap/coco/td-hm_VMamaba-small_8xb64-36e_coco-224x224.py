_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=2)

# optimizer
# custom_imports = dict(
#     imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
#     allow_failed_imports=False)
# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-8, betas=(0.9, 0.999)),
#     paramwise_cfg=dict(
#         # num_layers=24,
#         # layer_decay_rate=0.75,
#         custom_keys={
#             # 'backbone': dict(decay_multi=0.1),
#             'bias': dict(decay_multi=0.0),
#             'pos_embed': dict(decay_mult=0.0),
#             # 'relative_position_bias_table': dict(decay_mult=0.0),
#             'norm': dict(decay_mult=0.0),
#         },
#     ),
#     # constructor='LayerDecayOptimWrapperConstructor',
#     # clip_grad=dict(max_norm=1., norm_type=2),
# )
#
# # learning policy
# param_scheduler = [
#     dict(
#         type='LinearLR', begin=0, end=50, start_factor=5,
#         by_epoch=True),  # warm-up
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=210,
#         milestones=[180,200],
#         gamma=0.1,
#         by_epoch=True)
# ]
# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         # _delete_=True,
#         type='AdamW',
#         lr=0.0001,
#         betas=(0.9, 0.999),
#         weight_decay=0.05))
#
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=36,
#         by_epoch=True,
#         milestones=[27, 33],
#         gamma=0.1)
# ]

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]



# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=5))

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(224, 224), heatmap_size=(56, 56), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='Backbone_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="",
        # copied from classification/configs/vssm/vssm_small_224.yaml
        dims=96,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",  # v3_noz
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.3,
        norm_layer="ln2d",
    ),
    head=dict(
        type='HeatmapHead_VMamba',
        in_channels=384,
        out_channels=17,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=48,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

find_unused_parameters=True
