_base_ = ['./w8_weight_only.py']

numeric_optimization = dict(
    candidate_kind='w8a8',
    conditional_admission=True,
    stage_order=('calibrate', 'convert', 'train', 'profile', 'evaluate',
                 'latency', 'compare'),
    calibration=dict(
        source_candidate='full-s-v1', split='train2017', shuffle=False,
        worker_count=0, sample_count=512,
        artifact_binding='runtime-sha256-required'),
    quant_policy=dict(
        spec=dict(
            enabled=True, weight_bits=8, activation_bits=8,
            activation_scale='runtime-calibration-artifact',
            per_output_channel=True, symmetric=True)),
    train_envelope=dict(
        recovery='one-bounded-qat-or-distillation-run',
        requires_attributed_error=True, max_preliminary_ap_drop=0.3,
        resume_checkpoints=2),
)
