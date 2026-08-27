_base_ = ['../coco_s_v1_deterministic.py']

custom_imports = dict(
    imports=['mambapose_opt.numeric_conversion'], allow_failed_imports=False)
custom_hooks = [dict(type='NumericRuntimeHook', priority='VERY_HIGH')]

numeric_optimization = dict(
    schema_version=1, route='ssm-quant-pwl', candidate_kind='pwl',
    conditional_admission=True,
    pwl=dict(
             enabled_function='silu', source='module',
             roles=(
                 'backbone.layers.0.blocks.0.op.act',
                 'backbone.layers.1.blocks.0.op.act',
                 'backbone.layers.2.blocks.0.op.act',
                 'backbone.layers.2.blocks.1.op.act',
                 'backbone.layers.3.blocks.0.op.act'),
             domain=(-6.0, 6.0), segments=16,
             grid_points=4097, saturation='clamp', qat_form='differentiable'),
    attention_softmax='exact-floating',
    stage_order=('profile', 'evaluate', 'latency'),
)
