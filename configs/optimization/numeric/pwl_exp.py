_base_ = ['./pwl_silu.py']
numeric_optimization = dict(
    pwl=dict(enabled_function='exp', domain=(-4.0, 4.0)))
