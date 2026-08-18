# Copyright (c) OpenMMLab. All rights reserved.
from .dekr_head import DEKRHead
from .vis_head import VisPredictHead
from .yoloxpose_head import YOLOXPoseHead

OPTIONAL_IMPORT_ERRORS = {}

try:
    from .rtmo_head import RTMOHead
except ModuleNotFoundError as error:
    if error.name != 'mmdet' and not error.name.startswith('mmdet.'):
        raise
    OPTIONAL_IMPORT_ERRORS['RTMOHead'] = str(error)

__all__ = ['DEKRHead', 'VisPredictHead', 'YOLOXPoseHead']
if 'RTMOHead' in locals():
    __all__.append('RTMOHead')
