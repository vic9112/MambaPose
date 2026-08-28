_base_ = ['./_base_/paired_300ep.py']

formal_role = 'baseline'
formal_seed = 1
experiment_id = 'formal-stage-c-full-seed1'
work_dir = 'work_dirs/optimization/formal-stage-c/full-seed1'
randomness = dict(seed=1, deterministic=True)
train_dataloader = dict(sampler=dict(seed=1))
val_dataloader = dict(sampler=dict(seed=1))
test_dataloader = dict(sampler=dict(seed=1))
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='full')))
