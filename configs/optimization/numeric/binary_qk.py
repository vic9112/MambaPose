_base_ = ['../coco_s_v1_deterministic.py']

model = dict(head=dict(tokenpose_cfg=dict(qk_mode='binary')))
numeric_optimization = dict(
    schema_version=1, route='ssm-quant-pwl', candidate_kind='binary-qk',
    conditional_admission=True, head_layers=6, zero_sign=1,
    qk_ste=True, preserve_scale=True, softmax='floating', value='floating',
    attention_accumulation='floating', output_projection='floating',
    report='theoretical-changed-multiplies-separate-from-latency',
    stage_order=('smoke-stage-a', 'profile', 'evaluate', 'latency'),
    reference_scope='mechanism-inspired-sign-only-qk-preliminary',
    learnable_attention_bias=False, binaryattention_reproduction=False,
    train_envelope=dict(
        recovery='one-bounded-qat-self-distillation-after-stage-b-only',
        requires_stage_b_pass=True, requires_attributed_error=True,
        max_preliminary_ap_drop=0.3),
)
