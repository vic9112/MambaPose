from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from mambapose_opt.formal_checkpoint import (
    FileAuthority,
    FormalCheckpointError,
    TensorShape,
    load_formal_backbone_initialization,
    load_formal_tensor_checkpoint,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    path = tmp_path / 'weights.pth'
    torch.save({'weight': torch.arange(6, dtype=torch.float32).reshape(2, 3)}, path)
    authority = FileAuthority(str(tmp_path), 'weights.pth', _sha256(path))
    expected = {'weight': TensorShape((2, 3), 'torch.float32')}
    return path, authority, expected


def test_tensor_only_loader_accepts_exact_finite_state(tmp_path):
    path, authority, expected = _fixture(tmp_path)
    loaded = load_formal_tensor_checkpoint(path, authority, expected)
    assert tuple(loaded) == ('weight',)
    assert torch.equal(loaded['weight'], torch.arange(6).reshape(2, 3))


@pytest.mark.parametrize('mutation', ['path', 'sha', 'key', 'shape', 'dtype', 'finite'])
def test_tensor_only_loader_rejects_authority_or_tensor_drift(tmp_path, mutation):
    path, authority, expected = _fixture(tmp_path)
    if mutation == 'path':
        other = tmp_path / 'same.pth'
        other.write_bytes(path.read_bytes())
        path = other
    elif mutation == 'sha':
        authority = replace(authority, sha256='0' * 64)
    elif mutation == 'key':
        expected = {'other': TensorShape((2, 3), 'torch.float32')}
    elif mutation == 'shape':
        expected = {'weight': TensorShape((3, 2), 'torch.float32')}
    elif mutation == 'dtype':
        expected = {'weight': TensorShape((2, 3), 'torch.float64')}
    else:
        torch.save({'weight': torch.tensor([[float('nan')]])}, path)
        authority = replace(authority, sha256=_sha256(path))
        expected = {'weight': TensorShape((1, 1), 'torch.float32')}
    with pytest.raises(FormalCheckpointError):
        load_formal_tensor_checkpoint(path, authority, expected)


def test_tensor_only_loader_rejects_symlink_at_every_path_boundary(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    real = root / 'real.pth'
    torch.save({'weight': torch.zeros(1)}, real)
    alias = root / 'alias.pth'
    alias.symlink_to(real)
    authority = FileAuthority(str(root), 'alias.pth', _sha256(real))
    with pytest.raises(FormalCheckpointError, match='symlink'):
        load_formal_tensor_checkpoint(
            alias, authority, {'weight': TensorShape((1,), 'torch.float32')})


def test_tensor_only_loader_rejects_traversal_authority(tmp_path):
    path, authority, expected = _fixture(tmp_path)
    forged = replace(authority, path='../' + path.name)
    with pytest.raises(FormalCheckpointError, match='canonical'):
        load_formal_tensor_checkpoint(path, forged, expected)


def test_tensor_only_loader_rejects_hostile_pickle_without_execution(tmp_path):
    marker = tmp_path / 'executed'

    class Hostile:
        def __reduce__(self):
            return (Path.write_text, (marker, 'unsafe'))

    path = tmp_path / 'hostile.pth'
    torch.save({'weight': torch.zeros(1), 'payload': Hostile()}, path)
    authority = FileAuthority(str(tmp_path), path.name, _sha256(path))
    with pytest.raises(FormalCheckpointError, match='weights-only'):
        load_formal_tensor_checkpoint(
            path, authority, {'weight': TensorShape((1,), 'torch.float32')})
    assert not marker.exists()


def test_tensor_only_loader_rejects_non_tensor_recursive_leaf(tmp_path):
    path = tmp_path / 'metadata.pth'
    torch.save({'state_dict': {'weight': torch.zeros(1)}, 'note': 'unsafe'}, path)
    authority = FileAuthority(str(tmp_path), path.name, _sha256(path))
    with pytest.raises(FormalCheckpointError, match='tensor-only'):
        load_formal_tensor_checkpoint(
            path, authority, {'weight': TensorShape((1,), 'torch.float32')})


def test_tensor_only_loader_never_honors_force_unsafe_environment(tmp_path, monkeypatch):
    path, authority, expected = _fixture(tmp_path)
    monkeypatch.setenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    with pytest.raises(FormalCheckpointError, match='forbidden'):
        load_formal_tensor_checkpoint(path, authority, expected)


def test_backbone_initialization_uses_safe_model_projection(tmp_path):
    source = torch.nn.Linear(3, 2)
    path = tmp_path / 'upstream.pth'
    torch.save({
        'model': source.state_dict(),
        'epoch': 262,
        'optimizer': {'step': 1},
    }, path)
    authority = FileAuthority(str(tmp_path), path.name, _sha256(path))
    target = torch.nn.Linear(3, 2)
    report = load_formal_backbone_initialization(
        path, authority, target, minimum_compatible_tensors=2)
    assert report.compatible_tensors == 2
    assert all(torch.equal(target.state_dict()[key], source.state_dict()[key])
               for key in source.state_dict())


def test_backbone_initialization_rejects_hostile_bundle(tmp_path):
    marker = tmp_path / 'unsafe'

    class Hostile:
        def __reduce__(self):
            return (Path.write_text, (marker, 'executed'))

    path = tmp_path / 'hostile-upstream.pth'
    torch.save({'model': {'weight': torch.ones(1)}, 'payload': Hostile()}, path)
    authority = FileAuthority(str(tmp_path), path.name, _sha256(path))
    with pytest.raises(FormalCheckpointError, match='weights-only'):
        load_formal_backbone_initialization(
            path, authority, torch.nn.Linear(1, 1),
            minimum_compatible_tensors=1)
    assert not marker.exists()
