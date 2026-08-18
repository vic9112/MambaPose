from io import BytesIO
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
OLD_BUNDLED = ROOT / 'mmpose/models/backbones/Vim/causal-conv1d'


def test_bundled_causal_conv_source_is_rejected_as_api_incompatible():
    from tools.reproduction.fetch_causal_conv import source_has_mamba_110_api

    assert not source_has_mamba_110_api(OLD_BUNDLED)


def test_upstream_110_api_detection_requires_seq_idx_and_new_binding(tmp_path):
    from tools.reproduction.fetch_causal_conv import source_has_mamba_110_api

    (tmp_path / 'csrc').mkdir()
    (tmp_path / 'causal_conv1d').mkdir()
    (tmp_path / 'csrc/causal_conv1d.cpp').write_text(
        'causal_conv1d_fwd(x, weight, bias, seq_idx, activation);\n'
        'causal_conv1d_bwd(x, weight, bias, dout, seq_idx, dx, activation);\n')
    (tmp_path / 'causal_conv1d/__init__.py').write_text(
        '__version__ = "1.1.0"\n')
    assert source_has_mamba_110_api(tmp_path)


def test_patch_setup_replaces_legacy_architectures_with_sm120():
    from tools.reproduction.fetch_causal_conv import patch_setup_source

    source = '''
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")
'''
    patched = patch_setup_source(source)
    assert 'arch=compute_120,code=sm_120' in patched
    assert 'arch=compute_70' not in patched
    assert 'arch=compute_80' not in patched
    assert 'arch=compute_90' not in patched


def test_safe_extract_rejects_parent_path(tmp_path):
    from tools.reproduction.fetch_causal_conv import safe_extract_tar

    archive = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        info = tarfile.TarInfo('../escape')
        payload = b'bad'
        info.size = len(payload)
        bundle.addfile(info, BytesIO(payload))
    with pytest.raises(RuntimeError, match='unsafe archive member'):
        safe_extract_tar(archive, tmp_path / 'output')


def test_official_source_archive_pin_is_exact():
    from tools.reproduction.fetch_causal_conv import (
        CAUSAL_SOURCE_SHA256,
        CAUSAL_SOURCE_URL,
    )

    assert CAUSAL_SOURCE_URL.endswith('/refs/tags/v1.1.0.tar.gz')
    assert CAUSAL_SOURCE_SHA256 == (
        '2f1463cdcbf27c4b7fc4fa7bb89b0eccd4ea118da4e6c75d567c1ff746cbf4bc')
