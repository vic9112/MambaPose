_base_ = ['../coco_s_v1_deterministic.py']

model = dict(head=dict(tokenpose_cfg=dict(qk_mode='binary')))
numeric_optimization = dict(
    schema_version=1, route='ssm-quant-pwl', candidate_kind='binary-qk',
    conditional_admission=True, head_layers=6, zero_sign=1,
    qk_ste=True, preserve_scale=True, softmax='floating', value='floating',
    attention_accumulation='floating', output_projection='floating',
    report='theoretical-changed-multiplies-separate-from-latency',
    stage_order=('profile', 'evaluate', 'latency'),
    train_envelope=dict(
        recovery='one-bounded-qat-and-distillation-run',
        requires_attributed_error=True, max_preliminary_ap_drop=0.3),
)
