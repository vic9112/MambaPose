_base_ = ['../coco_s_v1_deterministic.py']

custom_imports = dict(
    imports=['mambapose_opt.numeric_conversion'], allow_failed_imports=False)
custom_hooks = [dict(type='NumericRuntimeHook', priority='VERY_HIGH')]

_vmamba_blocks = (
    'backbone.layers.0.blocks.0',
    'backbone.layers.1.blocks.0',
    'backbone.layers.2.blocks.0',
    'backbone.layers.2.blocks.1',
    'backbone.layers.3.blocks.0',
)
_backbone_roles = (
    'backbone.patch_embed.0', 'backbone.patch_embed.5',
    'backbone.layers.0.downsample.1',
    'backbone.layers.1.downsample.1',
    'backbone.layers.2.downsample.1',
) + tuple(
    f'{block}.{role}'
    for block in _vmamba_blocks
    for role in ('op.in_proj', 'op.conv2d', 'op.out_proj',
                 'mlp.fc1', 'mlp.fc2'))
_pif_roles = tuple(
    f'head.tokenpose.pose_interaction.{block}.mixer.{role}'
    for block in ('MambaBlock', 'MambaBlock2', 'Mamba_selfScanBlock')
    for role in ('in_proj', 'x_proj', 'dt_proj', 'out_proj'))
_attention_roles = tuple(
    f'head.tokenpose.transformer.layers.{layer}.{branch}.fn.fn.{role}'
    for layer in range(6)
    for branch, role in (
        ('0', 'to_qkv'), ('0', 'to_out.0'),
        ('1', 'net.0'), ('1', 'net.3')))

numeric_optimization = dict(
    schema_version=1,
    route='ssm-quant-pwl',
    candidate_kind='weight-only',
    simulation_only=True,
    integer_kernel_latency_claimed=False,
    stage_order=('convert', 'export', 'profile', 'evaluate', 'latency',
                 'compare'),
    quant_policy=dict(
        allow=(
            _backbone_roles + _pif_roles + _attention_roles
            + ('head.tokenpose.patch_to_embedding',
               'head.tokenpose.mlp_head.1')),
        deny=tuple(
            f'{block}.{role}'
            for block in _vmamba_blocks
            for role in ('norm', 'op.out_norm')) + (
                'head.tokenpose.pose_interaction.within_norm',
                'head.tokenpose.pose_interaction.within_norm_2',
                'head.tokenpose.mlp_head.0',
            ),
        spec=dict(enabled=True, weight_bits=8, activation_bits=None,
                  per_output_channel=True, symmetric=True)),
    precision_invariants=dict(
        selective_scan_state_accumulation='fp32',
        attention_softmax='floating', norms='floating'),
    runtime_envelope=dict(device='cuda:0', batch_size=1,
                          warmup_iterations=50, measured_iterations=200),
)

del _vmamba_blocks, _backbone_roles, _pif_roles, _attention_roles
