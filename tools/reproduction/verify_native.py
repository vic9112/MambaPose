#!/usr/bin/env python3
"""Run native/model CUDA gates and record admitted module artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(os.environ.get(
    'MAMBAPOSE_NATIVE_EVIDENCE_OUTPUT',
    REPO_ROOT / 'work_dirs/reproduction/evidence/native.json'))
MODULES = (
    'causal_conv1d_cuda',
    'selective_scan_cuda',
    'selective_scan_cuda_core',
    'selective_scan_cuda_ndstate',
    'selective_scan_cuda_oflex',
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    test_command = [
        sys.executable, '-m', 'pytest', '-q',
        'tests/test_reproduction/test_native_cuda.py',
        'tests/test_reproduction/test_model_cuda.py',
    ]
    result = subprocess.run(
        test_command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return result.returncode

    import torch
    import triton

    cuobjdump = (
        Path(triton.__file__).parent / 'backends/nvidia/bin/cuobjdump')
    artifacts = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        path = Path(module.__file__)
        listing = subprocess.run(
            [str(cuobjdump), '--list-elf', str(path)],
            check=True,
            capture_output=True,
            text=True).stdout.splitlines()
        cubins = [line.split(': ', 1)[-1] for line in listing
                  if '.cubin' in line]
        if not cubins or not all('.sm_120.cubin' in cubin for cubin in cubins):
            raise RuntimeError(f'{module_name} contains non-sm120 cubins')
        artifacts.append({
            'module': module_name,
            'path': str(path),
            'bytes': path.stat().st_size,
            'sha256': _sha256(path),
            'cubins': cubins,
        })
    evidence = {
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'torch_version': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(),
        'capability': list(torch.cuda.get_device_capability()),
        'pytest': {
            'command': test_command,
            'summary': result.stdout.splitlines()[-1],
        },
        'artifacts': artifacts,
    }
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEFAULT_OUTPUT.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    os.replace(temporary, DEFAULT_OUTPUT)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
