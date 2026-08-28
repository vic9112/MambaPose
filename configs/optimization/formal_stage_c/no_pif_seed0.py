_base_ = ['./_base_/paired_300ep.py']

formal_role = 'no_pif'
formal_seed = 0
experiment_id = 'formal-stage-c-no-pif-seed0'
work_dir = 'work_dirs/optimization/formal-stage-c/no-pif-seed0'
randomness = dict(seed=0, deterministic=True)
train_dataloader = dict(sampler=dict(seed=0))
val_dataloader = dict(sampler=dict(seed=0))
test_dataloader = dict(sampler=dict(seed=0))
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='disabled')))
