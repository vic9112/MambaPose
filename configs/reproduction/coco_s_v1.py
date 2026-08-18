_base_ = [
    '../body_2d_keypoint/tokenpose/'
    'mamba_tokenpose_T2_coco_256x192_300ep.py'
]

experiment_id = 'coco-s-v1'
paper_target = dict(
    dataset='coco',
    split='val2017',
    metric='AP',
    value=72.8,
    metrics=dict(AP=72.8, AP50=89.7, AP75=80.5, APM=69.4, APL=79.2,
                 AR=78.2),
    gflops=2.8)
randomness = dict(seed=0, deterministic=False)
model = dict(
    backbone=dict(
        depths=[1, 1, 2, 1],
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
