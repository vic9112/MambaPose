#!/usr/bin/env python3
"""Shared RTX 5090 native-extension build policy and entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping

_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from tools.reproduction.fetch_causal_conv import (
    CAUSAL_SOURCE_SHA256,
    CAUSAL_SOURCE_URL,
    prepare_source,
)


REPO_ROOT = _IMPORT_ROOT
DEFAULT_ENV_PREFIX = REPO_ROOT / '.venv'
WHEELHOUSE = REPO_ROOT / 'work_dirs/reproduction/wheelhouse'
BUILD_LOG_DIR = REPO_ROOT / 'work_dirs/reproduction/build_logs'
BUILD_MANIFEST = REPO_ROOT / 'work_dirs/reproduction/evidence/native-build.json'
MAMBA_SOURCE = REPO_ROOT / 'mmpose/models/backbones/Vim/mamba-1p1p1'
VMAMBA_SOURCE = (
    REPO_ROOT / 'mmpose/models/backbones/Vmamba/kernels/selective_scan')


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

    def prepend_path(variable: str, *paths: Path) -> None:
        existing = environment.get(variable, '')
        entries = [str(path) for path in paths]
        if existing:
            entries.append(existing)
        environment[variable] = os.pathsep.join(entries)

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
    # Conda's CUDA toolkit keeps target headers and libraries below
    # targets/x86_64-linux instead of CUDA_HOME/include and CUDA_HOME/lib64.
    # PyTorch's CUDA wheels also split library headers below
    # site-packages/nvidia/*/include. BuildExtension discovers neither layout.
    cuda_target = env_prefix / 'targets/x86_64-linux'
    nvidia_roots = sorted(
        env_prefix.glob('lib/python*/site-packages/nvidia'))
    nvidia_includes = sorted({
        path
        for root in nvidia_roots
        for path in root.glob('*/include')
        if path.is_dir()
    })
    nvidia_libraries = sorted({
        path
        for root in nvidia_roots
        for path in root.glob('*/lib')
        if path.is_dir()
    })
    prepend_path('CPATH', cuda_target / 'include', *nvidia_includes)
    prepend_path(
        'LIBRARY_PATH', cuda_target / 'lib', env_prefix / 'lib',
        *nvidia_libraries)
    prepend_path(
        'LD_LIBRARY_PATH', cuda_target / 'lib', env_prefix / 'lib',
        *nvidia_libraries)
    conda_cc = env_prefix / 'bin/x86_64-conda-linux-gnu-cc'
    conda_cxx = env_prefix / 'bin/x86_64-conda-linux-gnu-c++'
    if conda_cc.is_file():
        environment['CC'] = str(conda_cc)
    if conda_cxx.is_file():
        environment['CXX'] = str(conda_cxx)
    return environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _run_logged(
        command: list[str], cwd: Path, environment: Mapping[str, str],
        log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        log.write(f"cwd={cwd}\ncommand={json.dumps(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True)
    if result.returncode:
        tail = '\n'.join(
            log_path.read_text(encoding='utf-8', errors='replace')
            .splitlines()[-80:])
        raise RuntimeError(
            f'native build command failed with status {result.returncode}; '
            f'log={log_path}\n{tail}')


def _find_wheel(distribution_prefix: str) -> Path:
    normalized = distribution_prefix.replace('-', '_').lower()
    candidates = [
        path for path in WHEELHOUSE.glob('*.whl')
        if path.name.lower().replace('-', '_').startswith(normalized + '_')
        or path.name.lower().replace('-', '_').startswith(normalized + '-')
    ]
    if not candidates:
        raise RuntimeError(
            f'wheel for {distribution_prefix} not found in {WHEELHOUSE}')
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _build_and_install(
        distribution: str, source: Path, env_prefix: Path,
        environment: Mapping[str, str]) -> Path:
    python = str(env_prefix / 'bin/python')
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    _run_logged(
        [python, '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps',
         '--wheel-dir', str(WHEELHOUSE), str(source)],
        source,
        environment,
        BUILD_LOG_DIR / f'{distribution}-build.log')
    wheel = _find_wheel(distribution)
    _run_logged(
        [python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
         str(wheel)],
        REPO_ROOT,
        environment,
        BUILD_LOG_DIR / f'{distribution}-install.log')
    return wheel


def build_all(env_prefix: Path = DEFAULT_ENV_PREFIX) -> dict[str, object]:
    version = detect_nvcc_version(env_prefix)
    blackwell_gencode(version)
    environment = build_environment(env_prefix)
    causal_source = prepare_source()

    wheels: list[Path] = []
    wheels.append(_build_and_install(
        'causal_conv1d', causal_source, env_prefix, environment))
    wheels.append(_build_and_install(
        'mamba_ssm', MAMBA_SOURCE, env_prefix, environment))
    wheels.append(_build_and_install(
        'selective_scan', VMAMBA_SOURCE, env_prefix, environment))

    import torch

    manifest: dict[str, object] = {
        'built_at': datetime.now(timezone.utc).isoformat(),
        'python': str(env_prefix / 'bin/python'),
        'torch_version': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'torch_cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        'nvcc_version': list(version),
        'gencode': blackwell_gencode(version),
        'causal_source': {
            'url': CAUSAL_SOURCE_URL,
            'sha256': CAUSAL_SOURCE_SHA256,
            'path': str(causal_source.relative_to(REPO_ROOT)),
        },
        'wheels': [
            {
                'path': str(wheel.relative_to(REPO_ROOT)),
                'bytes': wheel.stat().st_size,
                'sha256': _sha256(wheel),
            }
            for wheel in wheels
        ],
    }
    BUILD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUILD_MANIFEST.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    os.replace(temporary, BUILD_MANIFEST)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-prefix', type=Path, default=DEFAULT_ENV_PREFIX)
    parser.add_argument('--show-policy', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        print(json.dumps(build_all(args.env_prefix), indent=2, sort_keys=True))
        return 0
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
