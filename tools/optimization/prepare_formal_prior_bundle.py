#!/usr/bin/env python3
"""Prepare the immutable Stage-B no-PIF authority for formal Stage C.

The checkpoint is copied and hashed as opaque bytes.  This standard-library
tool never imports Torch and never deserializes any artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Mapping, NamedTuple


SOURCE_COMMIT = 'a6adf6d84f9b51b133b0fca7272c51f728bfb7ac'
UNPRUNED_PARENT_SHA256 = (
    '28cd02405e58d619896a0430f91936684084ef7e3423d507c7a10b743759a5fb')
CANONICAL_DESTINATION = Path(
    '/home/vicchen/workspace/MambaPose/work_dirs/optimization/'
    'formal-stage-c-prior/no-pif-seed0')
DEFAULT_SOURCE_ROOT = Path(
    '/home/vicchen/workspace/MambaPose/.worktrees/'
    'algo-structural-runtime-frozen')
DEFAULT_AUDIT_REPORT = Path(
    '/home/vicchen/workspace/MambaPose/.superpowers/sdd/'
    '2026-08-27-mambapose-hardware-friendly-optimization-stage-ab/'
    'task-4-final-artifact-audit.md')
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGICAL_LINK = (
    DEFAULT_REPOSITORY_ROOT / 'work_dirs/optimization/prior-stage-b')

_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_CANDIDATE_ROOT = Path(
    'work_dirs/optimization/structural-pif/no-pif-s-v1/0')
_SOURCE_ENTRIES = {
    'train': (_CANDIDATE_ROOT / 'train/train.json',
              Path('artifacts/train.json')),
    'runtime_metadata': (
        _CANDIDATE_ROOT / 'train/runtime-metadata.json',
        Path('artifacts/runtime-metadata.json')),
    'pruned_checkpoint': (
        _CANDIDATE_ROOT / 'train/best_route2.pth',
        Path('artifacts/pruned-runtime.pth')),
    'profile': (_CANDIDATE_ROOT / 'profile/profile.json',
                Path('artifacts/profile.json')),
    'evaluate': (_CANDIDATE_ROOT / 'evaluate/evaluate.json',
                 Path('artifacts/evaluate.json')),
    'latency': (_CANDIDATE_ROOT / 'latency/latency.json',
                Path('artifacts/latency.json')),
    'source_config': (
        Path('configs/optimization/structural/no_pif_pruned.py'),
        Path('source/no_pif_pruned.py')),
    'candidate_manifest': (
        Path('optimization/candidates.json'),
        Path('source/candidates.json')),
    'coco_authority': (
        Path('optimization/coco_val2017_authority.json'),
        Path('source/coco_val2017_authority.json')),
    'parent_config': (
        Path('configs/reproduction/ablations/coco_s_v1_no_pif.py'),
        Path('source/coco_s_v1_no_pif.py')),
}
_TRACKED_ENTRIES = frozenset({
    'source_config', 'candidate_manifest', 'coco_authority', 'parent_config'})
PRODUCTION_HASHES = MappingProxyType({
    'train': 'ea276713f214acce1949798445203e7e05026dd4ed7d6e76e77cffde35041819',
    'runtime_metadata': '99e75ff3a4862f08eba3a7c75e586617aa1438efa8f164a73da093ffe5b672ce',
    'pruned_checkpoint': '5797ceaffbf7d369d8eaf8f47b548a79bd43d66dd603933571fa3b88ff28e3db',
    'profile': 'cb0c933291c28bc341cc5bb62b956c14a76255825321bc0291d4e688fa7aa7a3',
    'evaluate': '9bcf326417cb0b9ffbf78deb01d19ac09525d9deb491c93189e8b769df0f3bf5',
    'latency': '9e7448a6b9ab96eb625c08fe0d6e5e9c3933349da2098cb91ab633faca41b98e',
    'source_config': 'e2a84676bb489ea4a9bb3f4a68a965afb7d2168efa013880732db6c818139df2',
    'candidate_manifest': '1081b546af0a9f9c95e9ca5e4fffca6f9609f38bf111db6bf47c0c1f5dca3d76',
    'coco_authority': '5d5945c35f9bf59d2ff537c088dad3119ae9d7e5ead0b51a641b20667c8c9edb',
    'parent_config': '706eaa316934d89a446510651d3d7dc3b34278436fa8caa47bdf3c228d1209e1',
    'independent_audit': 'b7382dbaca7294477990c2463bf642f3959762a7b9e54d626e46e45d8bb5f361',
})


class PriorBundleError(RuntimeError):
    """Raised when an approved prior authority cannot be reproduced."""


class PriorBundleResult(NamedTuple):
    status: str
    destination: Path
    logical_link: Path
    bundle_sha256: str


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise PriorBundleError(f'authority source is not a regular file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ['git', *arguments], cwd=root, check=check,
            capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PriorBundleError('cannot inspect structural source Git state') from error


def _verify_source(root: Path, expected_commit: str) -> None:
    if not root.is_dir() or root.is_symlink():
        raise PriorBundleError('clean detached source root is unavailable')
    actual = _git(root, 'rev-parse', 'HEAD').stdout.strip()
    symbolic = _git(root, 'symbolic-ref', '-q', 'HEAD', check=False)
    status = _git(
        root, 'status', '--porcelain', '--untracked-files=all').stdout
    allowed_untracked = {
        f'?? {relative.as_posix()}'
        for key, (relative, _destination) in _SOURCE_ENTRIES.items()
        if key not in _TRACKED_ENTRIES
    }
    unexpected = {
        line for line in status.splitlines()
        if line and line not in allowed_untracked}
    if (
            actual != expected_commit
            or symbolic.returncode == 0
            or unexpected):
        raise PriorBundleError(
            'formal prior requires the exact clean detached source')


def _verify_sha256_map(value: Mapping[str, str]) -> None:
    expected_keys = set(_SOURCE_ENTRIES) | {'independent_audit'}
    if set(value) != expected_keys:
        raise PriorBundleError('expected hash inventory has wrong fields')
    if any(
            not isinstance(item, str) or not _SHA256.fullmatch(item)
            for item in value.values()):
        raise PriorBundleError('expected hash inventory contains an invalid hash')


def _read_json(path: Path, field: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise PriorBundleError(f'{field} is not valid JSON') from error
    if not isinstance(value, dict):
        raise PriorBundleError(f'{field} must be a JSON object')
    return value


def _nested(value: Mapping[str, object], field: str, *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise PriorBundleError(f'{field} is missing {".".join(keys)}')
        current = current[key]
    return current


def _require_equal(
        value: Mapping[str, object], field: str, expected: object,
        *keys: str) -> None:
    if _nested(value, field, *keys) != expected:
        raise PriorBundleError(
            f'{field} has inconsistent {".".join(keys)}')


def _validate_production_artifact_semantics(source_root: Path) -> None:
    """Close the transitive identity graph before copying approved bytes."""
    root = source_root
    documents = {
        name: _read_json(root / _SOURCE_ENTRIES[name][0], name)
        for name in (
            'train', 'runtime_metadata', 'profile', 'evaluate', 'latency',
            'candidate_manifest')}
    checkpoint_path = (
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/'
        'train/best_route2.pth')
    metadata_path = (
        'work_dirs/optimization/structural-pif/no-pif-s-v1/0/'
        'train/runtime-metadata.json')
    config_path = 'configs/optimization/structural/no_pif_pruned.py'
    parent_config_path = 'configs/reproduction/ablations/coco_s_v1_no_pif.py'
    manifest_path = 'optimization/candidates.json'
    authority_path = 'optimization/coco_val2017_authority.json'
    runtime = {
        'checkpoint': {
            'path': checkpoint_path,
            'sha256': PRODUCTION_HASHES['pruned_checkpoint']},
        'config': {
            'path': config_path,
            'sha256': PRODUCTION_HASHES['source_config']},
        'metadata': {
            'path': metadata_path,
            'sha256': PRODUCTION_HASHES['runtime_metadata']},
        'table': None,
        'transform': 'prune-disabled-pif-v1',
    }
    source = {
        'authority_path': authority_path,
        'authority_sha256': PRODUCTION_HASHES['coco_authority'],
        'config_path': parent_config_path,
        'config_sha256': PRODUCTION_HASHES['parent_config'],
        'git_commit': SOURCE_COMMIT,
        'manifest_path': manifest_path,
        'manifest_sha256': PRODUCTION_HASHES['candidate_manifest'],
    }
    parent = {
        'checkpoint': (
            'work_dirs/reproduction/runs/coco-s-v1-no-pif/'
            'best_coco_AP_epoch_300.pth'),
        'checkpoint_sha256': UNPRUNED_PARENT_SHA256,
        'config': parent_config_path,
    }

    train = documents['train']
    for keys, expected in (
            (('candidate_id',), 'no-pif-s-v1'),
            (('stage',), 'train'),
            (('result', 'route'), 'structural-pif'),
            (('result', 'runtime'), runtime),
            (('result', 'source'), source),
            (('result', 'parent'), {**parent, 'role': 'trained-parent'}),
            (('result', 'protocol', 'operation'),
             'deterministic-prune-export-zero-training'),
            (('result', 'protocol', 'seed'), 0)):
        _require_equal(train, 'train', expected, *keys)

    metadata = documents['runtime_metadata']
    for keys, expected in (
            (('candidate_id',), 'no-pif-s-v1'),
            (('route',), 'structural-pif'),
            (('parent_checkpoint_sha256',), UNPRUNED_PARENT_SHA256),
            (('runtime_checkpoint_sha256',),
             PRODUCTION_HASHES['pruned_checkpoint']),
            (('transform',), 'prune-disabled-pif-v1')):
        _require_equal(metadata, 'runtime_metadata', expected, *keys)

    profile = documents['profile']
    for keys, expected in (
            (('candidate',), 'no-pif-s-v1'),
            (('checkpoint',), checkpoint_path),
            (('checkpoint_sha256',), PRODUCTION_HASHES['pruned_checkpoint']),
            (('config',), config_path),
            (('git_commit',), SOURCE_COMMIT),
            (('runtime',), runtime),
            (('parent',), parent)):
        _require_equal(profile, 'profile', expected, *keys)

    for name in ('evaluate', 'latency'):
        document = documents[name]
        for keys, expected in (
                (('candidate_id',), 'no-pif-s-v1'),
                (('stage',), name),
                (('result', 'route'), 'structural-pif'),
                (('result', 'runtime'), runtime),
                (('result', 'source'), source)):
            _require_equal(document, name, expected, *keys)
    evaluate = documents['evaluate']
    for mode in ('flip', 'no_flip'):
        _require_equal(
            evaluate, 'evaluate', PRODUCTION_HASHES['pruned_checkpoint'],
            'result', 'modes', mode, 'provenance', 'checkpoint_sha256')
        _require_equal(
            evaluate, 'evaluate', SOURCE_COMMIT,
            'result', 'modes', mode, 'provenance', 'git_commit')
    latency = documents['latency']
    _require_equal(
        latency, 'latency', PRODUCTION_HASHES['pruned_checkpoint'],
        'result', 'provenance', 'checkpoint_sha256')
    _require_equal(
        latency, 'latency', SOURCE_COMMIT,
        'result', 'provenance', 'git_commit')
    _require_equal(latency, 'latency', parent, 'result', 'parent')

    candidates = _nested(
        documents['candidate_manifest'], 'candidate_manifest', 'candidates')
    if not isinstance(candidates, list):
        raise PriorBundleError('candidate_manifest candidates must be a list')
    selected = [
        row for row in candidates
        if isinstance(row, dict) and row.get('id') == 'no-pif-s-v1']
    if len(selected) != 1:
        raise PriorBundleError(
            'candidate_manifest must contain one no-pif-s-v1 row')
    row = selected[0]
    for key, expected in (
            ('checkpoint_sha256', UNPRUNED_PARENT_SHA256),
            ('config', parent_config_path),
            ('route', 'structural-pif'),
            ('seed', 0)):
        if row.get(key) != expected:
            raise PriorBundleError(
                f'candidate_manifest no-pif-s-v1 {key} mismatch')
    features = row.get('features')
    if not isinstance(features, dict) or (
            features.get('deployment_config') != config_path
            or features.get('runtime_checkpoint') != checkpoint_path):
        raise PriorBundleError(
            'candidate_manifest no-pif-s-v1 runtime binding mismatch')


def _collect_entries(
        source_root: Path, audit_report: Path,
        expected_commit: str, expected_hashes: Mapping[str, str],
        ) -> tuple[dict[str, dict[str, int | str]], dict[str, Path]]:
    _verify_sha256_map(expected_hashes)
    if expected_commit == SOURCE_COMMIT:
        if dict(expected_hashes) != dict(PRODUCTION_HASHES):
            raise PriorBundleError(
                'production prior requires the predeclared approved inventory')
        _validate_production_artifact_semantics(source_root)
    entries: dict[str, dict[str, int | str]] = {}
    sources: dict[str, Path] = {}
    for name, (source_relative, destination_relative) in _SOURCE_ENTRIES.items():
        source = source_root / source_relative
        digest = _sha256_file(source)
        if digest != expected_hashes[name]:
            raise PriorBundleError(f'{name} hash mismatch')
        if name in _TRACKED_ENTRIES:
            blob = _git(
                source_root, 'show',
                f'{expected_commit}:{source_relative.as_posix()}').stdout.encode()
            if hashlib.sha256(blob).hexdigest() != digest or blob != source.read_bytes():
                raise PriorBundleError(f'{name} tracked source mismatch')
        entries[name] = {
            'path': destination_relative.as_posix(),
            'sha256': digest,
            'bytes': source.stat().st_size,
        }
        sources[name] = source
    audit_digest = _sha256_file(audit_report)
    if audit_digest != expected_hashes['independent_audit']:
        raise PriorBundleError('independent_audit hash mismatch')
    audit_destination = Path('audit/task-4-final-artifact-audit.md')
    entries['independent_audit'] = {
        'path': audit_destination.as_posix(),
        'sha256': audit_digest,
        'bytes': audit_report.stat().st_size,
    }
    sources['independent_audit'] = audit_report
    return entries, sources


def _bundle_document(
        source_commit: str, entries: Mapping[str, Mapping[str, int | str]],
        unpruned_parent_sha256: str) -> dict[str, object]:
    if not _SHA256.fullmatch(unpruned_parent_sha256):
        raise PriorBundleError('unpruned parent SHA-256 is invalid')
    return {
        'schema_version': 1,
        'kind': 'mambapose-formal-stage-c-prior-no-pif-seed0',
        'source_commit': source_commit,
        'unpruned_parent_checkpoint_sha256': unpruned_parent_sha256,
        'entries': dict(sorted(entries.items())),
    }


def _canonical_bytes(document: Mapping[str, object]) -> bytes:
    return (json.dumps(
        document, sort_keys=True, indent=2,
        ensure_ascii=False) + '\n').encode('utf-8')


def _validate_existing(destination: Path, expected: bytes) -> str:
    manifest = destination / 'bundle.json'
    if not destination.is_dir() or destination.is_symlink():
        raise PriorBundleError('existing bundle is not an approved directory')
    if not manifest.is_file() or manifest.read_bytes() != expected:
        raise PriorBundleError('existing bundle does not match authority')
    document = json.loads(expected)
    allowed = {'bundle.json'}
    allowed_directories: set[str] = set()
    for record in document['entries'].values():
        relative = Path(record['path'])
        allowed.add(relative.as_posix())
        parent = relative.parent
        while parent != Path('.'):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
        candidate = destination / relative
        if (
                _sha256_file(candidate) != record['sha256']
                or candidate.stat().st_size != record['bytes']):
            raise PriorBundleError('existing bundle file mismatch')
    actual = set()
    actual_directories = set()
    for path in destination.rglob('*'):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            raise PriorBundleError('existing bundle contains an unexpected symlink')
        if path.is_file():
            actual.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise PriorBundleError('existing bundle contains an unexpected entry')
    if actual != allowed:
        raise PriorBundleError('existing bundle contains unexpected files')
    if actual_directories != allowed_directories:
        raise PriorBundleError('existing bundle contains unexpected directories')
    for path in (destination, *destination.rglob('*')):
        if path.stat().st_mode & stat.S_IWUSR:
            raise PriorBundleError('existing bundle is not read-only')
    return _sha256_file(manifest)


def _reject_parent_symlinks(path: Path, field: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current = current / part
        if current.is_symlink():
            raise PriorBundleError(f'{field} parent must not be a symlink')


def _preflight_link(link: Path, destination: Path) -> None:
    if os.path.lexists(link):
        if not link.is_symlink():
            raise PriorBundleError('logical link is not a symlink')
        try:
            target = link.resolve(strict=True)
        except OSError as error:
            raise PriorBundleError('logical link target is unavailable') from error
        if target != destination.resolve(strict=False):
            raise PriorBundleError('logical link target drift')


def _install_link(link: Path, destination: Path) -> None:
    if os.path.lexists(link):
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f'.{link.name}.{os.getpid()}.tmp')
    try:
        temporary.symlink_to(destination)
        os.replace(temporary, link)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob('*'), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _prepare_prior_bundle(
        *, source_root: Path | str, destination: Path | str,
        logical_link: Path | str, audit_report: Path | str,
        expected_source_commit: str, expected_hashes: Mapping[str, str],
        unpruned_parent_sha256: str) -> PriorBundleResult:
    source = Path(source_root).absolute()
    target = Path(destination).absolute()
    link = Path(logical_link).absolute()
    audit = Path(audit_report).absolute()
    if target == source or target == link or source in target.parents:
        raise PriorBundleError('bundle paths overlap source or link authority')
    _reject_parent_symlinks(target, 'bundle destination')
    _reject_parent_symlinks(link, 'logical link')
    _verify_source(source, expected_source_commit)
    entries, sources = _collect_entries(
        source, audit, expected_source_commit, expected_hashes)
    document = _bundle_document(
        expected_source_commit, entries, unpruned_parent_sha256)
    manifest_bytes = _canonical_bytes(document)
    _preflight_link(link, target)

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        bundle_sha = _validate_existing(target, manifest_bytes)
        _install_link(link, target)
        return PriorBundleResult('current', target, link, bundle_sha)

    staging = Path(tempfile.mkdtemp(
        prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent))
    installed = False
    try:
        for name, record in entries.items():
            destination_file = staging / str(record['path'])
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[name], destination_file)
            if (
                    destination_file.stat().st_size != record['bytes']
                    or _sha256_file(destination_file) != record['sha256']):
                raise PriorBundleError(f'{name} copied bundle hash mismatch')
        bundle_manifest = staging / 'bundle.json'
        bundle_manifest.write_bytes(manifest_bytes)
        _make_read_only(staging)
        os.rename(staging, target)
        installed = True
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)
    _install_link(link, target)
    bundle_sha = _validate_existing(target, manifest_bytes)
    return PriorBundleResult('written', target, link, bundle_sha)


def _check_prior_bundle(
        *, source_root: Path | str, destination: Path | str,
        logical_link: Path | str, audit_report: Path | str,
        expected_source_commit: str, expected_hashes: Mapping[str, str],
        unpruned_parent_sha256: str) -> PriorBundleResult:
    """Read-only validation of the source, bundle, and logical link."""
    source = Path(source_root).absolute()
    target = Path(destination).absolute()
    link = Path(logical_link).absolute()
    _reject_parent_symlinks(target, 'bundle destination')
    _reject_parent_symlinks(link, 'logical link')
    _verify_source(source, expected_source_commit)
    entries, _sources = _collect_entries(
        source, Path(audit_report).absolute(), expected_source_commit,
        expected_hashes)
    expected = _canonical_bytes(_bundle_document(
        expected_source_commit, entries, unpruned_parent_sha256))
    _preflight_link(link, target)
    if not target.exists():
        raise PriorBundleError('existing bundle is missing')
    bundle_sha = _validate_existing(target, expected)
    return PriorBundleResult('current', target, link, bundle_sha)


def prepare_prior_bundle() -> PriorBundleResult:
    """Prepare only the predeclared, independently audited authority."""
    return _prepare_prior_bundle(
        source_root=DEFAULT_SOURCE_ROOT,
        destination=CANONICAL_DESTINATION,
        logical_link=DEFAULT_LOGICAL_LINK,
        audit_report=DEFAULT_AUDIT_REPORT,
        expected_source_commit=SOURCE_COMMIT,
        expected_hashes=PRODUCTION_HASHES,
        unpruned_parent_sha256=UNPRUNED_PARENT_SHA256)


def check_prior_bundle() -> PriorBundleResult:
    """Read-only validation of only the predeclared authority."""
    return _check_prior_bundle(
        source_root=DEFAULT_SOURCE_ROOT,
        destination=CANONICAL_DESTINATION,
        logical_link=DEFAULT_LOGICAL_LINK,
        audit_report=DEFAULT_AUDIT_REPORT,
        expected_source_commit=SOURCE_COMMIT,
        expected_hashes=PRODUCTION_HASHES,
        unpruned_parent_sha256=UNPRUNED_PARENT_SHA256)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check is not None:
        expected_manifest = DEFAULT_LOGICAL_LINK / 'bundle.json'
        if args.check.absolute() != expected_manifest.absolute():
            raise PriorBundleError(
                '--check must name the declared logical bundle manifest')
        result = check_prior_bundle()
    else:
        result = prepare_prior_bundle()
    print(json.dumps({
        'status': result.status,
        'destination': str(result.destination),
        'logical_link': str(result.logical_link),
        'bundle_sha256': result.bundle_sha256,
        'torch_imported': 'torch' in sys.modules,
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
