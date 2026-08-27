_base_ = ['./pwl_silu.py']
numeric_optimization = dict(
    pwl=dict(enabled_function='softplus', domain=(-8.0, 8.0)))
