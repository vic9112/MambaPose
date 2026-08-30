_base_ = ['../../reproduction/ablations/coco_s_v1_no_pif.py']

custom_imports = dict(
    imports=['mambapose_opt.numeric_conversion'], allow_failed_imports=False)
custom_hooks = [dict(type='NumericRuntimeHook', priority='VERY_HIGH')]

numeric_optimization = dict(
    schema_version=1,
    route='ssm-quant-pwl',
    candidate_kind='pwl-combined',
    conditional_admission=True,
    calibration=dict(
        artifact_schema_version=3,
        split='train2017',
        samples=512,
        exact_input_roles=True,
        source_candidate_id='no-pif-s-v1'),
    pwl=dict(
        candidate_id='no-pif-pwl-softplus-s-v1',
        enabled_function='softplus',
        source='ss2d-transition',
        roles=(
            'backbone.layers.0.blocks.0.op',
            'backbone.layers.1.blocks.0.op',
            'backbone.layers.2.blocks.0.op',
            'backbone.layers.2.blocks.1.op',
            'backbone.layers.3.blocks.0.op'),
        domain=(-8.0, 8.0),
        segments=16,
        grid_points=4097,
        saturation='continuous-asymptotic-tail-v1',
        qat_form='differentiable',
        selection_policy='observed-range-max-then-mean-v1'),
    comparator=dict(candidate_id='full-s-v1', metric='coco-val2017-AP'),
    claim_limits=dict(
        combined_accuracy_measured=False,
        isolated_deltas_additive=False,
        interaction_effect_measured=False,
        hardware_latency_claimed=False,
        fpga_speedup_claimed=False),
    attention_softmax='exact-floating',
    stage_order=(
        'calibrate', 'convert', 'smoke-stage-a', 'profile', 'evaluate',
        'compare', 'latency'))
