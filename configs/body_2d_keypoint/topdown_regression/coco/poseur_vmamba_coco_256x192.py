# log_level = 'INFO'
# load_from = None
# resume_from = None
# dist_params = dict(backend='nccl')
# workflow = [('train', 1)]
# checkpoint_config = dict(interval=10)
# evaluation = dict(interval=25, metric='mAP', key_indicator='AP', rle_score=True)
#
# optimizer = dict(
#     type='AdamW',
#     lr=1e-3,
#     weight_decay=1e-4,
#     paramwise_cfg = dict(
#         custom_keys={
#             # 'backbone': dict(lr_mult=0.1),
#             'sampling_offsets': dict(lr_mult=0.1),
#             'reference_points': dict(lr_mult=0.1),
#             # 'query_embed': dict(lr_mult=0.5, decay_mult=1.0),
#         },
#     )
# )
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[255, 310])
# total_epochs = 325
#
# log_config = dict(
#     interval=50, hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='TensorboardLoggerHook'),
#     ])
#
# channel_cfg = dict(
#     num_output_channels=17,
#     dataset_joints=17,
#     dataset_channel=[
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
#     ],
#     inference_channel=[
#         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
#     ])
#
# emb_dim = 256
#
# # model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     type='Poseur',
#     pretrained='https://download.openmmlab.com/mmpose/'
#     'pretrain_models/hrnet_w32-36af842e.pth',
#     backbone=dict(
#         type='HRNet',
#         norm_cfg = norm_cfg,
#         in_channels=3,
#         extra=dict(
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(4, ),
#                 num_channels=(64, )),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 block='BASIC',
#                 num_blocks=(4, 4),
#                 num_channels=(32, 64)),
#             stage3=dict(
#                 num_modules=4,
#                 num_branches=3,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4),
#                 num_channels=(32, 64, 128)),
#             stage4=dict(
#                 num_modules=3,
#                 num_branches=4,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4, 4),
#                 num_channels=(32, 64, 128, 256),
#                 multiscale_output=True,
#                 )),
#     ),
#     neck=dict(
#         type='ChannelMapper',
#         in_channels=[32, 64, 128, 256],
#         kernel_size=1,
#         out_channels=emb_dim,
#         act_cfg=None,
#         norm_cfg=dict(type='GN', num_groups=32),
#     ),
#     keypoint_head=dict(
#         type='PoseurHead',
#         in_channels=512,
#         num_queries=17,
#         num_reg_fcs=2,
#         num_joints=channel_cfg['num_output_channels'],
#         with_box_refine=True,
#         loss_coord_enc=dict(type='RLELoss_poseur', use_target_weight=True),
#         loss_coord_dec=dict(type='RLELoss_poseur', use_target_weight=True),
#         # loss_coord_dec=dict(type='L1Loss', use_target_weight=True, loss_weight=5),
#         loss_hp_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=10),
#         # loss_coord_keypoint=dict(type='L1Loss', use_target_weight=True, loss_weight=1),
#         positional_encoding=dict(
#             type='SinePositionalEncoding',
#             num_feats=emb_dim//2,
#             normalize=True,
#             offset=-0.5),
#         transformer=dict(
#             type='PoseurTransformer',
#             query_pose_emb = True,
#             embed_dims = emb_dim,
#             encoder=dict(
#                 type='DetrTransformerEncoder_zero_layer',
#                 num_layers=0,
#                 transformerlayers=dict(
#                     type='BaseTransformerLayer',
#                     ffn_cfgs = dict(
#                         embed_dims=emb_dim,
#                         ),
#                     attn_cfgs=dict(
#                         type='MultiScaleDeformableAttention',
#                         num_levels=4,
#                         num_points=4,
#                         embed_dims=emb_dim),
#
#                     feedforward_channels=1024,
#                     ffn_dropout=0.1,
#                     operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
#             decoder=dict(
#                 type='DeformableDetrTransformerDecoder',
#                 num_layers=6,
#                 return_intermediate=True,
#                 transformerlayers=dict(
#                     type='DetrTransformerDecoderLayer_grouped',
#                     ffn_cfgs = dict(
#                         embed_dims=emb_dim,
#                         ),
#                     attn_cfgs=[
#                         dict(
#                             type='MultiheadAttention',
#                             embed_dims=emb_dim,
#                             num_heads=8,
#                             dropout=0.1),
#                         dict(
#                             type='MultiScaleDeformableAttention_post_value',
#                             num_levels=4,
#                             num_points=4,
#                             embed_dims=emb_dim)
#                     ],
#                     feedforward_channels=1024,
#                     ffn_dropout=0.1,
#                     operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
#                                      'ffn', 'norm')))),
#         as_two_stage=True,
#         use_heatmap_loss=False,
#     ),
#     train_cfg=dict(image_size=[192, 256]),
#     test_cfg = dict(
#         image_size=[192, 256],
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11)
# )
#
# data_cfg = dict(
#     image_size=[192, 256],
#     heatmap_size=[48, 64],
#     num_output_channels=channel_cfg['num_output_channels'],
#     num_joints=channel_cfg['dataset_joints'],
#     dataset_channel=channel_cfg['dataset_channel'],
#     inference_channel=channel_cfg['inference_channel'],
#     soft_nms=False,
#     # use_nms=False,
#     nms_thr=1.0,
#     oks_thr=0.9,
#     vis_thr=0.2,
#     det_bbox_thr=0.0,
#     # use_gt_bbox=True,
#     # bbox_file='',
#     use_gt_bbox=False,
#     bbox_file='data/coco/person_detection_results/'
#     'COCO_val2017_detections_AP_H_56_person.json',
#
# )
#
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='TopDownGetBboxCenterScale', padding=1.25),
#     dict(type='TopDownRandomFlip', flip_prob=0.5),
#     dict(
#         type='TopDownHalfBodyTransform',
#         num_joints_half_body=8,
#         prob_half_body=0.3),
#     dict(
#         type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
#     dict(type='TopDownAffine'),
#     dict(type='ToTensor'),
#     dict(
#         type='NormalizeTensor',
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]),
#     # dict(
#     #     type='TopDownGenerateTarget',
#     #     kernel=[(11, 11), (9, 9), (7, 7), (5, 5)],
#     #     encoding='Megvii'),
#     dict(
#         target_type='wo_mask',
#         type='TopDownGenerateCoordAndHeatMapTarget',
#         encoding='MSRA',
#         sigma=2),
#     dict(
#         type='Collect',
#         keys=['img', 'coord_target', 'coord_target_weight', 'hp_target', 'hp_target_weight'],
#         meta_keys=[
#             'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
#             'rotation', 'bbox_score', 'flip_pairs'
#         ]),
# ]
#
# val_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='TopDownGetBboxCenterScale', padding=1.25),
#     dict(type='TopDownAffine'),
#     dict(type='ToTensor'),
#     dict(
#         type='NormalizeTensor',
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]),
#     dict(
#         type='Collect',
#         keys=[
#             'img',
#         ],
#         meta_keys=[
#             'image_file', 'center', 'scale', 'rotation', 'bbox_score',
#             'flip_pairs'
#         ]),
# ]
#
# test_pipeline = val_pipeline
#
# data_root = 'data/coco'
# data = dict(
#     samples_per_gpu=8,
#     # samples_per_gpu=64,
#     workers_per_gpu=8,
#     val_dataloader=dict(samples_per_gpu=32),
#     test_dataloader=dict(samples_per_gpu=32),
#     train=dict(
#         type='TopDownCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
#         img_prefix=f'{data_root}/train2017/',
#         # ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
#         # img_prefix=f'{data_root}/val2017/',
#         data_cfg=data_cfg,
#         pipeline=train_pipeline),
#     val=dict(
#         type='TopDownCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
#         img_prefix=f'{data_root}/val2017/',
#         data_cfg=data_cfg,
#         pipeline=val_pipeline),
#     test=dict(
#         type='TopDownCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
#         img_prefix=f'{data_root}/val2017/',
#         data_cfg=data_cfg,
#         pipeline=val_pipeline),
# )
#
# fp16 = dict(loss_scale='dynamic')


_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=40, val_interval=5)

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=5e-5, type='AdamW', weight_decay=1e-5),
    paramwise_cfg = dict(
        custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
            # 'query_embed': dict(lr_mult=0.5, decay_mult=1.0),
        },
    )
)

# learning policy
param_scheduler = [
    # dict(
    #     type='LinearLR', begin=0, end=1000, start_factor=0.01,
    #     by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        milestones=[30],
        gamma=0.1,
        by_epoch=True)
]
optimizer_config = dict(grad_clip=None)
# automatically scaling LR based on the actual training batch size
# auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(type='RegressionLabel', input_size=(192, 256))

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

emb_dim = 256

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
        pretrained="vssm_small_0229_ckpt_epoch_222.pth",
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
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
    ),
    head=dict(
        type='PoseurHead',
        in_channels=512,
        num_queries=17,
        num_reg_fcs=2,
        decoder=codec,
        num_joints=channel_cfg['num_output_channels'],
        with_box_refine=True,
        loss_coord_enc=dict(type='RLELoss_poseur', use_target_weight=True),
        loss_coord_dec=dict(type='RLELoss_poseur', use_target_weight=True),
        loss_hp_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=emb_dim//2,
            normalize=True,
            offset=-0.5),
        transformer=dict(
            type='PoseurTransformer',
            query_pose_emb = True,
            embed_dims = emb_dim,
            encoder_config=dict(
                # type='DeformableDetrTransformerEncoder',
                num_layers=1,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=4,
                        num_points=4,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0))),
            decoder_config=dict(
                # type='DeformableDetrTransformerDecoder',
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True),
                    cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=2048, ffn_drop=0.1)),
                return_intermediate=True),
        as_two_stage=True,
    ),
    train_cfg=dict(image_size=[192, 256]),
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,
    ))
)

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=48,
    num_workers=8,
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
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=f'{data_root}person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

find_unused_parameters=True
load_from = "work_dirs/poseur_vmamba_coco_256x192_9_18/epoch_170.pth"
