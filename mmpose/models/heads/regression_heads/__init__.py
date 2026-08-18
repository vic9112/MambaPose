# Copyright (c) OpenMMLab. All rights reserved.
from .dsnt_head import DSNTHead
from .integral_regression_head import IntegralRegressionHead
from .motion_regression_head import MotionRegressionHead
from .regression_head import RegressionHead
from .rle_head import RLEHead
from .temporal_regression_head import TemporalRegressionHead
from .trajectory_regression_head import TrajectoryRegressionHead
from .poseur_head import PoseurHead
from .positional_encoding import SinePositionalEncoding

OPTIONAL_IMPORT_ERRORS = {}

try:
    from .transformer_poseur import PoseurTransformer
except ModuleNotFoundError as error:
    if error.name != 'mmcv._ext' and not error.name.startswith('mmcv.ops'):
        raise
    OPTIONAL_IMPORT_ERRORS['PoseurTransformer'] = str(error)

__all__ = [
    'RegressionHead', 'IntegralRegressionHead', 'DSNTHead', 'RLEHead',
    'TemporalRegressionHead', 'TrajectoryRegressionHead',
    'MotionRegressionHead', 'PoseurHead', 'SinePositionalEncoding'
]
if 'PoseurTransformer' in locals():
    __all__.append('PoseurTransformer')
