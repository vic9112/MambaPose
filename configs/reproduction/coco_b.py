_base_ = [
    '../body_2d_keypoint/tokenpose/'
    'mamba_tokenpose_T_coco_256x192_300ep.py'
]

experiment_id = 'coco-b'
paper_target = dict(
    dataset='coco',
    split='val2017',
    metric='AP',
    value=75.0,
    metrics=dict(AP=75.0, AP50=90.5, AP75=82.7, APM=71.3, APL=81.5,
                 AR=80.1),
    gflops=5.2)
randomness = dict(seed=0, deterministic=False)
model = dict(
    backbone=dict(
        depths=[2, 2, 5, 2],
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
