_base_ = [
    '../body_2d_keypoint/topdown_heatmap/crowdpose/'
    'mamba_tokenpose_T2_crowdpose_256x192_300ep.py'
]

experiment_id = 'crowdpose-s-v1'
paper_target = dict(
    dataset='crowdpose',
    split='test',
    metric='AP',
    value=65.6,
    metrics=dict(AP=65.6, AR=75.2),
    gflops=2.8)
randomness = dict(seed=0, deterministic=False)
model = dict(
    backbone=dict(
        depths=[1, 1, 2, 1],
        pretrained='pretrained/vssm_tiny_0230_ckpt_epoch_262.pth'),
    head=dict(tokenpose_cfg=dict(pif_mode='full')))
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best='crowdpose/AP',
        rule='greater'))

data_root = 'data/crowdpose/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        bbox_file=(
            'data/crowdpose/annotations/'
            'det_for_crowd_test_0.1_0.5.json')))
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file='data/crowdpose/annotations/mmpose_crowdpose_test.json')
test_evaluator = val_evaluator

