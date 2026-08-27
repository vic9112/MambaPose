_base_ = ['./w8_weight_only.py']

numeric_optimization = dict(
    candidate_kind='w8a8',
    conditional_admission=True,
    stage_order=('calibrate', 'convert', 'profile', 'evaluate', 'latency'),
    calibration=dict(
        source_candidate='full-s-v1', split='train2017', shuffle=False,
        worker_count=0, sample_count=512,
        artifact_binding='runtime-sha256-required'),
    quant_policy=dict(
        allow=(
            'backbone.layers.0.blocks.0.op.in_proj',
            'backbone.layers.0.blocks.0.op.out_proj',
            'backbone.layers.1.blocks.0.op.in_proj',
            'backbone.layers.1.blocks.0.op.out_proj',
            'backbone.layers.2.blocks.0.op.in_proj',
            'backbone.layers.2.blocks.0.op.out_proj',
            'backbone.layers.2.blocks.1.op.in_proj',
            'backbone.layers.2.blocks.1.op.out_proj',
            'backbone.layers.3.blocks.0.op.in_proj',
            'backbone.layers.3.blocks.0.op.out_proj',
        ) + tuple(
            f'head.tokenpose.transformer.layers.{layer}.0.fn.fn.to_qkv'
            for layer in range(6)) + ('head.tokenpose.mlp_head.1',),
        activation_observers={
            **{
                f'backbone.layers.{layer}.blocks.{block}.op.{projection}':
                f'backbone.layers.{layer}.blocks.{block}.op.{projection}.input'
                for layer, block in ((0, 0), (1, 0), (2, 0), (2, 1), (3, 0))
                for projection in ('in_proj', 'out_proj')
            },
            **{
                f'head.tokenpose.transformer.layers.{layer}.0.fn.fn.to_qkv':
                f'head.tokenpose.transformer.layers.{layer}.0.fn.fn.to_qkv.input'
                for layer in range(6)
            },
            'head.tokenpose.mlp_head.1':
                'head.tokenpose.mlp_head.1.input',
        },
        spec=dict(
            enabled=True, weight_bits=8, activation_bits=8,
            activation_scale='runtime-calibration-artifact',
            per_output_channel=True, symmetric=True)),
    train_envelope=dict(
        recovery='one-bounded-qat-or-distillation-run',
        requires_attributed_error=True, max_preliminary_ap_drop=0.3,
        resume_checkpoints=2),
)
