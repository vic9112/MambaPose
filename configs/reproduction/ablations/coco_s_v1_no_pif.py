_base_ = ['../coco_s_v1.py']

experiment_id = 'coco-s-v1-no-pif'
paper_target = dict(
    dataset='coco', split='val2017', metric='AP', value=72.6, gflops=2.7)
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='disabled')))

