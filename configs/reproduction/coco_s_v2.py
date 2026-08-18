_base_ = [
    '../body_2d_keypoint/tokenpose/'
    'mamba_tokenpose_T3_coco_256x192_300ep.py'
]

experiment_id = 'coco-s-v2'
paper_target = dict(
    dataset='coco',
    split='val2017',
    metric='AP',
    value=74.2,
    metrics=dict(AP=74.2, AP50=90.5, AP75=82.0, APM=70.9, APL=80.6,
                 AR=79.6),
    gflops=4.0)
randomness = dict(seed=0, deterministic=False)
model = dict(
    backbone=dict(
        depths=[1, 2, 3, 2],
        pretrained='pretrained/vssm_tiny_0230_ckpt_epoch_262.pth',
        pretrained_strict=True,
        minimum_pretrained_tensors=100),
    head=dict(tokenpose_cfg=dict(pif_mode='full')))
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best='coco/AP',
        rule='greater'))
