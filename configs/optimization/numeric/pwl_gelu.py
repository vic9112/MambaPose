_base_ = ['./pwl_silu.py']
numeric_optimization = dict(
    pwl=dict(enabled_function='gelu', domain=(-5.0, 5.0)))
