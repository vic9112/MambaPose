_base_ = ['../crowdpose_s_v1.py']

experiment_id = 'crowdpose-s-v1-no-pif'
paper_target = dict(
    dataset='crowdpose', split='test', metric='AP', value=65.3,
    reported_gflops=5.1)
model = dict(head=dict(tokenpose_cfg=dict(pif_mode='disabled')))

