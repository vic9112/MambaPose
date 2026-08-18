#!/usr/bin/env python3
"""Fetch and patch the exact causal-conv1d source required by MambaPose."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import tarfile
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
CAUSAL_SOURCE_URL = (
    'https://github.com/Dao-AILab/causal-conv1d/'
    'archive/refs/tags/v1.1.0.tar.gz')
CAUSAL_SOURCE_SHA256 = (
    '2f1463cdcbf27c4b7fc4fa7bb89b0eccd4ea118da4e6c75d567c1ff746cbf4bc')
DEFAULT_CACHE = REPO_ROOT / 'work_dirs/reproduction/downloads'
DEFAULT_SOURCE_PARENT = REPO_ROOT / 'work_dirs/reproduction/sources'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_has_mamba_110_api(source_root: Path) -> bool:
    version_path = source_root / 'causal_conv1d/__init__.py'
    binding_path = source_root / 'csrc/causal_conv1d.cpp'
    if not version_path.is_file() or not binding_path.is_file():
        return False
    version = version_path.read_text(encoding='utf-8')
    binding = binding_path.read_text(encoding='utf-8')
    return (
        re.search(r'__version__\s*=\s*["\']1\.1\.0["\']', version)
        is not None
        and 'causal_conv1d_fwd' in binding
        and 'causal_conv1d_bwd' in binding
        and 'seq_idx' in binding)


def patch_setup_source(source: str) -> str:
    legacy_architectures = re.compile(
        r'(?ms)^    cc_flag\.append\("-gencode"\)\n'
        r'^    cc_flag\.append\("arch=compute_70,code=sm_70"\)\n'
        r'^    cc_flag\.append\("-gencode"\)\n'
        r'^    cc_flag\.append\("arch=compute_80,code=sm_80"\)\n'
        r'^    if bare_metal_version >= Version\("11\.8"\):\n'
        r'^        cc_flag\.append\("-gencode"\)\n'
        r'^        cc_flag\.append\("arch=compute_90,code=sm_90"\)\n')
    replacement = (
        '    cc_flag.extend(["-gencode", '
        '"arch=compute_120,code=sm_120"])\n')
    patched, replacements = legacy_architectures.subn(replacement, source)
    if replacements != 1:
        raise RuntimeError(
            'Expected exactly one legacy CUDA architecture block in '
            f'causal-conv1d setup.py, found {replacements}')
    patched = patched.replace(
        'if bare_metal_version < Version("11.6"):',
        'if bare_metal_version < Version("12.8"):')
    patched = patched.replace(
        'causal_conv1d is only supported on CUDA 11.6 and above.',
        'MambaPose causal_conv1d requires CUDA 12.8 or newer for sm_120.')
    return patched


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if (target != destination_root
                    and destination_root not in target.parents):
                raise RuntimeError(f'unsafe archive member: {member.name}')
            if member.issym() or member.islnk():
                raise RuntimeError(f'unsafe archive member link: {member.name}')
        bundle.extractall(destination)


def download_verified(
        url: str = CAUSAL_SOURCE_URL,
        expected_sha256: str = CAUSAL_SOURCE_SHA256,
        cache_dir: Path = DEFAULT_CACHE) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / 'causal-conv1d-v1.1.0.tar.gz'
    if archive.is_file() and sha256_file(archive) == expected_sha256:
        return archive

    partial = archive.with_suffix(archive.suffix + '.part')
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = offset > 0 and response.status == 206
        mode = 'ab' if resumed else 'wb'
        with partial.open(mode) as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)

    actual_sha256 = sha256_file(partial)
    if actual_sha256 != expected_sha256:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        quarantine = partial.with_name(partial.name + f'.bad.{timestamp}')
        os.replace(partial, quarantine)
        raise RuntimeError(
            f'causal-conv1d archive SHA-256 mismatch: {actual_sha256}; '
            f'quarantined at {quarantine}')
    os.replace(partial, archive)
    return archive


def prepare_source(
        cache_dir: Path = DEFAULT_CACHE,
        source_parent: Path = DEFAULT_SOURCE_PARENT) -> Path:
    archive = download_verified(cache_dir=cache_dir)
    source_parent.mkdir(parents=True, exist_ok=True)
    source_root = source_parent / 'causal-conv1d-1.1.0'
    setup_path = source_root / 'setup.py'
    if source_has_mamba_110_api(source_root) and setup_path.is_file():
        setup_source = setup_path.read_text(encoding='utf-8')
        if ('arch=compute_120,code=sm_120' in setup_source
                and 'arch=compute_70' not in setup_source):
            return source_root
    if source_root.exists():
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        os.replace(source_root, source_root.with_name(
            source_root.name + f'.invalid.{timestamp}'))

    safe_extract_tar(archive, source_parent)
    if not source_has_mamba_110_api(source_root):
        raise RuntimeError('official causal-conv1d v1.1.0 API audit failed')
    setup_path.write_text(
        patch_setup_source(setup_path.read_text(encoding='utf-8')),
        encoding='utf-8')
    return source_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        '--source-parent', type=Path, default=DEFAULT_SOURCE_PARENT)
    args = parser.parse_args()
    print(prepare_source(args.cache_dir, args.source_parent))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
