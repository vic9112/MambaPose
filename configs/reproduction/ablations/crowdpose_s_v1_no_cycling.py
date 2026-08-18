_base_ = ['../crowdpose_s_v1.py']

experiment_id = 'crowdpose-s-v1-no-cycling'
paper_target = dict(
    dataset='crowdpose', split='test', metric='AP', value=65.49)
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='no_cycling')))
