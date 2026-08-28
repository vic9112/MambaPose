_base_ = ['./_base_/paired_300ep.py']

formal_role = 'no_pif'
formal_seed = 2
experiment_id = 'formal-stage-c-no-pif-seed2'
work_dir = 'work_dirs/optimization/formal-stage-c/no-pif-seed2'
randomness = dict(seed=2, deterministic=True)
train_dataloader = dict(sampler=dict(seed=2))
val_dataloader = dict(sampler=dict(seed=2))
test_dataloader = dict(sampler=dict(seed=2))
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='disabled')))
