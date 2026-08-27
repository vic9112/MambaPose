_base_ = ['./coco_s_v1_deterministic.py']

experiment_id = 'coco-s-v1-no-pif-seed0-deterministic'
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='disabled')))
