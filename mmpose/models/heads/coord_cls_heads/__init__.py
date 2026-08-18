# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .rtmw_head import RTMWHead
from .simcc_head import SimCCHead

OPTIONAL_IMPORT_ERRORS = {}

try:
    from .simba_head import SimbaHead
except ModuleNotFoundError as error:
    if error.name != 'mamba_ssm' and not error.name.startswith('mamba_ssm.'):
        raise
    OPTIONAL_IMPORT_ERRORS['SimbaHead'] = str(error)

__all__ = ['SimCCHead', 'RTMCCHead', 'RTMWHead']
if 'SimbaHead' in locals():
    __all__.append('SimbaHead')
