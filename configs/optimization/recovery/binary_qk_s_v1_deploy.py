_base_ = ['../coco_s_v1_deterministic.py']

# The exported student is a normal pose estimator.  Its checkpoint layout is
# unchanged; this mode switch is the complete inference-time Binary Q/K bind.
model = dict(
    backbone=dict(pretrained=None),
    head=dict(tokenpose_cfg=dict(qk_mode='binary_scaled')),
)
load_from = None
resume = False
