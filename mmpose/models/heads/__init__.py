# Copyright (c) OpenMMLab. All rights reserved.
from .base_head import BaseHead
from .coord_cls_heads import RTMCCHead, RTMWHead, SimCCHead
from .heatmap_heads import (AssociativeEmbeddingHead, CIDHead, CPMHead,
                            HeatmapHead, InternetHead, MSPNHead, ViPNASHead)
from .hybrid_heads import DEKRHead, VisPredictHead
from .regression_heads import (DSNTHead, IntegralRegressionHead,
                               MotionRegressionHead, RegressionHead, RLEHead,
                               TemporalRegressionHead,
                               TrajectoryRegressionHead)

OPTIONAL_IMPORT_ERRORS = {}

try:
    from .coord_cls_heads import SimbaHead
except ImportError as error:
    OPTIONAL_IMPORT_ERRORS['SimbaHead'] = str(error)

try:
    from .transformer_heads import EDPoseHead
except ModuleNotFoundError as error:
    if error.name != 'mmcv._ext' and not error.name.startswith('mmcv.ops'):
        raise
    OPTIONAL_IMPORT_ERRORS['EDPoseHead'] = str(error)

try:
    from .hybrid_heads import RTMOHead
except ImportError as error:
    OPTIONAL_IMPORT_ERRORS['RTMOHead'] = str(error)

__all__ = [
    'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'RegressionHead', 'IntegralRegressionHead', 'SimCCHead', 'RLEHead',
    'DSNTHead', 'AssociativeEmbeddingHead', 'DEKRHead', 'VisPredictHead',
    'CIDHead', 'RTMCCHead', 'TemporalRegressionHead',
    'TrajectoryRegressionHead', 'MotionRegressionHead', 'InternetHead',
    'RTMWHead'
]
for optional_name in ('EDPoseHead', 'RTMOHead', 'SimbaHead'):
    if optional_name in locals():
        __all__.append(optional_name)
