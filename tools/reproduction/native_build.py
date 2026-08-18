#!/usr/bin/env python3
"""Shared RTX 5090 native-extension build policy and entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PREFIX = REPO_ROOT / '.venv'


def blackwell_gencode(cuda_version: tuple[int, int]) -> list[str]:
    """Return the only architecture flags admitted for this workstation."""
    if cuda_version < (12, 8):
        rendered = '.'.join(str(part) for part in cuda_version)
        raise RuntimeError(
            f'CUDA 12.8 or newer is required for sm_120; got {rendered}')
    return ['-gencode', 'arch=compute_120,code=sm_120']


def detect_nvcc_version(cuda_home: Path) -> tuple[int, int]:
    result = subprocess.run(
        [str(cuda_home / 'bin/nvcc'), '--version'],
        check=True,
        capture_output=True,
        text=True,
        timeout=30)
    match = re.search(r'release\s+(\d+)\.(\d+)', result.stdout)
    if match is None:
        raise RuntimeError(f'Cannot parse nvcc version from: {result.stdout}')
    return int(match.group(1)), int(match.group(2))


def build_environment(
        env_prefix: Path,
        base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an isolated, source-build-only environment."""
    environment = dict(os.environ if base is None else base)
    environment.pop('MAMBA_FORCE_CXX11_ABI', None)
    environment.pop('FORCE_CXX11_ABI', None)
    environment.update({
        'PYTHONNOUSERSITE': '1',
        'CUDA_HOME': str(env_prefix),
        'TORCH_CUDA_ARCH_LIST': '12.0',
        'MAMBA_FORCE_BUILD': 'TRUE',
        'CAUSAL_CONV1D_FORCE_BUILD': 'TRUE',
        'MAX_JOBS': environment.get('MAX_JOBS', '4'),
        'PATH': f"{env_prefix / 'bin'}:{environment.get('PATH', '')}",
    })
    conda_cc = env_prefix / 'bin/x86_64-conda-linux-gnu-cc'
    conda_cxx = env_prefix / 'bin/x86_64-conda-linux-gnu-c++'
    if conda_cc.is_file():
        environment['CC'] = str(conda_cc)
    if conda_cxx.is_file():
        environment['CXX'] = str(conda_cxx)
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-prefix', type=Path, default=DEFAULT_ENV_PREFIX)
    parser.add_argument('--show-policy', action='store_true')
    args = parser.parse_args()
    version = detect_nvcc_version(args.env_prefix)
    policy = {
        'cuda_version': list(version),
        'gencode': blackwell_gencode(version),
        'environment': {
            key: value
            for key, value in build_environment(args.env_prefix).items()
            if key in {
                'CAUSAL_CONV1D_FORCE_BUILD', 'CUDA_HOME', 'MAMBA_FORCE_BUILD',
                'MAX_JOBS', 'PYTHONNOUSERSITE', 'TORCH_CUDA_ARCH_LIST'
            }
        },
    }
    if args.show_policy:
        print(json.dumps(policy, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
