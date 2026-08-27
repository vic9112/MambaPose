_base_ = ['./pwl_silu.py']
numeric_optimization = dict(
    pwl=dict(
        enabled_function='exp', source='ss2d-transition',
        roles=(
            'backbone.layers.0.blocks.0.op',
            'backbone.layers.1.blocks.0.op',
            'backbone.layers.2.blocks.0.op',
            'backbone.layers.2.blocks.1.op',
            'backbone.layers.3.blocks.0.op'),
        domain=(-4.0, 4.0)))
