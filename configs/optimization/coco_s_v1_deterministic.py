_base_ = ['../reproduction/coco_s_v1.py']

custom_imports = dict(
    imports=['mambapose_opt.determinism'], allow_failed_imports=False)
randomness = dict(seed=0, deterministic=True)

_worker = dict(type='mambapose_seed_worker', base_seed=0)
train_dataloader = dict(
    num_workers=2, persistent_workers=False, worker_init_fn=_worker)
val_dataloader = dict(
    num_workers=2, persistent_workers=False, worker_init_fn=_worker)
test_dataloader = dict(
    num_workers=2, persistent_workers=False, worker_init_fn=_worker)
