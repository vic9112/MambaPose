_base_ = ['./_base_/paired_300ep.py']

formal_role = 'baseline'
formal_seed = 0
experiment_id = 'formal-stage-c-full-seed0'
work_dir = 'work_dirs/optimization/formal-stage-c/full-seed0'
randomness = dict(seed=0, deterministic=True)
train_dataloader = dict(sampler=dict(seed=0))
val_dataloader = dict(sampler=dict(seed=0))
test_dataloader = dict(sampler=dict(seed=0))
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='full')))
