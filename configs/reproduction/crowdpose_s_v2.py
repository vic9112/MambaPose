_base_ = [
    '../body_2d_keypoint/topdown_heatmap/crowdpose/'
    'mamba_tokenpose_T3_crowdpose_256x192_300ep.py'
]

experiment_id = 'crowdpose-s-v2'
paper_target = dict(
    dataset='crowdpose',
    split='test',
    metric='AP',
    value=67.0,
    metrics=dict(AP=67.0, AR=77.0),
    gflops=4.0)
randomness = dict(seed=0, deterministic=False)
model = dict(
    backbone=dict(
        # Paper Table II overrides the repository's conflicting [1,2,3,1].
        depths=[1, 2, 3, 2],
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

