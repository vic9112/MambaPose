_base_ = ['./_base_/paired_300ep.py']

formal_role = 'baseline'
formal_seed = 3
experiment_id = 'formal-stage-c-full-seed3'
work_dir = 'work_dirs/optimization/formal-stage-c/full-seed3'
randomness = dict(seed=3, deterministic=True)
train_dataloader = dict(sampler=dict(seed=3))
val_dataloader = dict(sampler=dict(seed=3))
test_dataloader = dict(sampler=dict(seed=3))
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='full')))
