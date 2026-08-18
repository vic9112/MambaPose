# Copyright (c) OpenMMLab. All rights reserved.
from .ae_head import AssociativeEmbeddingHead
from .cid_head import CIDHead
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .internet_head import InternetHead
from .mspn_head import MSPNHead
from .vipnas_head import ViPNASHead
from .heatmap_head_mamba import HeatmapHead_Mamba
from .heatmap_head_Vmamba import HeatmapHead_VMamba

OPTIONAL_IMPORT_ERRORS = {}

try:
    from .mamba_token_head import MambaTokenHead
except ModuleNotFoundError as error:
    if error.name != 'mamba_ssm' and not error.name.startswith('mamba_ssm.'):
        raise
    OPTIONAL_IMPORT_ERRORS['MambaTokenHead'] = str(error)

__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'AssociativeEmbeddingHead', 'CIDHead', 'InternetHead',
    'HeatmapHead_Mamba', 'HeatmapHead_VMamba'
]
if 'MambaTokenHead' in locals():
    __all__.append('MambaTokenHead')
