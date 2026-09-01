_base_ = ['./pwl_silu.py']
numeric_optimization = dict(
    pwl=dict(
        candidate_id='pwl-gelu-s-v1',
        enabled_function='gelu', source='module',
        roles=(
            'backbone.layers.0.blocks.0.mlp.act',
            'backbone.layers.1.blocks.0.mlp.act',
            'backbone.layers.2.blocks.0.mlp.act',
            'backbone.layers.2.blocks.1.mlp.act',
            'backbone.layers.3.blocks.0.mlp.act'),
        domain=(-5.0, 5.0)))
