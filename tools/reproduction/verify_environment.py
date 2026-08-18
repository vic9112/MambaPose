#!/usr/bin/env python3
"""Verify and record the isolated RTX 5090 reproduction environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(os.environ.get(
    'MAMBAPOSE_EVIDENCE_OUTPUT',
    REPO_ROOT / 'work_dirs/reproduction/evidence/environment.json'))


def _command(*args: str) -> str:
    result = subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def _nvcc_version() -> str:
    output = _command('nvcc', '--version')
    match = re.search(r'release\s+(\d+\.\d+)', output)
    if match is None:
        raise RuntimeError(f'Cannot parse nvcc version from: {output}')
    return match.group(1)


def collect() -> dict[str, Any]:
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot access the CUDA device')

    device_index = torch.cuda.current_device()
    lhs = torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4)
    rhs = torch.eye(4, device='cuda', dtype=torch.float32)
    cuda_smoke_checksum = float((lhs @ rhs).sum().cpu())
    driver_line = _command(
        'nvidia-smi', '--query-gpu=driver_version',
        '--format=csv,noheader').splitlines()[device_index]
    return {
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'cuda_runtime': torch.version.cuda,
        'nvcc_version': _nvcc_version(),
        'driver_version': driver_line.strip(),
        'device_index': device_index,
        'device_name': torch.cuda.get_device_name(device_index),
        'device_capability': list(torch.cuda.get_device_capability(device_index)),
        'cuda_smoke_checksum': cuda_smoke_checksum,
        'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        'gcc_version': _command('gcc', '-dumpfullversion'),
        'python_no_user_site': os.environ.get('PYTHONNOUSERSITE'),
    }


def validate(evidence: dict[str, Any]) -> None:
    errors: list[str] = []
    if not evidence['python_version'].startswith('3.11.'):
        errors.append(f"Python must be 3.11, got {evidence['python_version']}")
    if not evidence['torch_version'].startswith('2.7.1+cu128'):
        errors.append(f"Torch must be 2.7.1+cu128, got {evidence['torch_version']}")
    if not evidence['torchvision_version'].startswith('0.22.1+cu128'):
        errors.append(
            'torchvision must be 0.22.1+cu128, got '
            f"{evidence['torchvision_version']}")
    if evidence['cuda_runtime'] != '12.8':
        errors.append(f"Torch CUDA runtime must be 12.8, got {evidence['cuda_runtime']}")
    if tuple(int(part) for part in evidence['nvcc_version'].split('.')) < (12, 8):
        errors.append(f"nvcc must be >=12.8, got {evidence['nvcc_version']}")
    if evidence['device_capability'] != [12, 0]:
        errors.append(
            'GPU compute capability must be [12, 0], got '
            f"{evidence['device_capability']}")
    if evidence['python_no_user_site'] != '1':
        errors.append('PYTHONNOUSERSITE must be exactly 1')
    if errors:
        raise RuntimeError('; '.join(errors))


def write_atomic(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    os.replace(temporary, path)


def main() -> int:
    evidence = collect()
    validate(evidence)
    write_atomic(DEFAULT_OUTPUT, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
