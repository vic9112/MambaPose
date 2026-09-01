_base_ = ['./_base_/paired_300ep.py']

formal_role = 'baseline'
formal_seed = 4
experiment_id = 'formal-stage-c-full-seed4'
work_dir = 'work_dirs/optimization/formal-stage-c/full-seed4'
randomness = dict(seed=4, deterministic=True)
train_dataloader = dict(sampler=dict(seed=4))
val_dataloader = dict(sampler=dict(seed=4))
test_dataloader = dict(sampler=dict(seed=4))
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='full')))
