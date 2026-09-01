_base_ = ['../../../reproduction/coco_s_v1.py']

custom_imports = dict(
    imports=['mambapose_opt.determinism'], allow_failed_imports=False)

formal_protocol = dict(
    epochs=300,
    worker_count=2,
    persistent_workers=False,
    per_device_batch_size=128,
    world_size=1,
    accumulation_steps=1,
    effective_batch_size=128,
    primary_seeds=[0, 1, 2],
    conditional_seeds=[3, 4],
)

randomness = dict(seed=0, deterministic=True)
train_cfg = dict(max_epochs=300, val_interval=5)

_worker = dict(type='mambapose_seed_worker')
train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=False,
    worker_init_fn=_worker,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=True, seed=0),
)
val_dataloader = dict(
    num_workers=2,
    persistent_workers=False,
    worker_init_fn=_worker,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False, seed=0),
)
test_dataloader = dict(
    num_workers=2,
    persistent_workers=False,
    worker_init_fn=_worker,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False, seed=0),
)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best='coco/AP',
        rule='greater'))
