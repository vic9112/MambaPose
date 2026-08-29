_base_ = ['../coco_s_v1_deterministic.py']

# One bounded recovery is 20% of the original 300-epoch S-V1 training.
train_cfg = dict(max_epochs=60, val_interval=5)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),
    dict(
        type='MultiStepLR', begin=0, end=60, milestones=[40, 52],
        gamma=0.1, by_epoch=True),
]

model = dict(
    _delete_=True,
    type='BinaryQKSelfDistiller',
    base_model_config='configs/optimization/coco_s_v1_deterministic.py',
    checkpoint=(
        'work_dirs/reproduction/runs/coco-s-v1/'
        'best_coco_AP_epoch_300.pth'),
    checkpoint_sha256=(
        'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2'),
    distill_weight=0.25,
)
load_from = None
resume = False

default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=3, save_last=True,
        save_best='coco/AP', rule='greater'))

binary_qk_recovery = dict(
    schema_version=1,
    qk_mode='binary_scaled',
    teacher_qk_mode='float',
    teacher_frozen=True,
    supervised_loss='heatmap-mse-with-target-weight',
    distillation_target='final-heatmap-mse',
    schedule='bounded-60-epoch-recovery',
    schedule_extensible=True,
    learnable_attention_bias=False,
)
