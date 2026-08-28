_base_ = ['./_base_/paired_300ep.py']

formal_role = 'no_pif'
formal_seed = 4
experiment_id = 'formal-stage-c-no-pif-seed4'
work_dir = 'work_dirs/optimization/formal-stage-c/no-pif-seed4'
randomness = dict(seed=4, deterministic=True)
train_dataloader = dict(sampler=dict(seed=4))
val_dataloader = dict(sampler=dict(seed=4))
test_dataloader = dict(sampler=dict(seed=4))
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='disabled')))
