"""Commit-addressed source and policy authority for Route 3 artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
from typing import Any, Mapping

from .schema import CandidateSpec, parse_candidate_manifest
from .source import clean_git_commit
from .evaluation import resolve_shared_asset_exposure


class NumericSourceError(ValueError):
    """Raised when a Route 3 artifact cannot prove immutable source."""


_SHA = re.compile(r'^[0-9a-f]{64}$')
_COMMIT = re.compile(r'^[0-9a-f]{40}$')
_FIELDS = {
    'git_commit', 'manifest_path', 'manifest_sha256', 'candidate_id',
    'candidate_row_sha256', 'config_path', 'config_sha256',
    'checkpoint_path', 'checkpoint_sha256', 'policy_path', 'policy_sha256',
    'authority_path', 'authority_sha256'}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _blob(root: Path, commit: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            ['git', 'show', f'{commit}:{relative}'], cwd=root, check=True,
            capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise NumericSourceError(
            f'numeric source is not tracked at {commit}: {relative}') from error


def _relative(root: Path, path: Path, label: str) -> str:
    candidate = path if path.is_absolute() else root / path
    lexical = candidate.absolute()
    try:
        relative = lexical.relative_to(root.absolute())
    except ValueError as error:
        raise NumericSourceError(f'{label} escapes repository') from error
    if any(part in {'.', '..'} for part in relative.parts):
        raise NumericSourceError(f'{label} path is unsafe')
    cursor = root
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        if cursor.is_symlink():
            if not _approved_reproduction_link(
                    root, relative, cursor, index):
                raise NumericSourceError(
                    f'{label} path must not use unapproved symlinks')
    return relative.as_posix()


def _approved_reproduction_link(
        root: Path, relative: Path, _link: Path, index: int) -> bool:
    if tuple(relative.parts[:index + 1]) != ('work_dirs', 'reproduction'):
        return False
    try:
        common_value = subprocess.check_output(
            ['git', 'rev-parse', '--git-common-dir'], cwd=root,
            text=True).strip()
        common = Path(common_value)
        if not common.is_absolute():
            common = root / common
        common = common.resolve(strict=True)
        checkout = common.parent.resolve(strict=True)
        if common.name != '.git' or root.resolve(strict=True) == checkout:
            return False
        resolve_shared_asset_exposure(
            root, checkout, Path('work_dirs/reproduction'),
            label='numeric checkpoint')
        return True
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


def _candidate_row(blob: bytes, identifier: str) -> tuple[CandidateSpec, str]:
    try:
        decoded = json.loads(blob)
        candidates = parse_candidate_manifest(decoded)
        raw = [row for row in decoded['candidates'] if row.get('id') == identifier]
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise NumericSourceError(f'numeric source manifest is invalid: {error}') from error
    selected = tuple(row for row in candidates if row.id == identifier)
    if len(selected) != 1 or len(raw) != 1:
        raise NumericSourceError('numeric candidate must occur exactly once')
    encoded = json.dumps(
        raw[0], sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return selected[0], hashlib.sha256(encoded).hexdigest()


def resolve_numeric_file(
        repository_root: Path, path: Path, label: str) -> Path:
    """Resolve a safe lexical file, admitting only the canonical asset link."""
    root = Path(repository_root).resolve(strict=True)
    relative = _relative(root, path, label)
    selected = root / relative
    if not selected.is_file():
        raise NumericSourceError(f'{label} is missing: {relative}')
    return selected


def validate_numeric_config_closure(
        repository_root: Path, config_path: Path, *, git_commit: str,
        ) -> tuple[dict[str, str], ...]:
    """Reconstruct and bind every Python config inherited at one commit."""
    root = Path(repository_root).resolve(strict=True)
    if not isinstance(git_commit, str) or not _COMMIT.fullmatch(git_commit):
        raise NumericSourceError('numeric config commit is invalid')
    start = _relative(root, config_path, 'config')
    visited: dict[str, str] = {}
    active: set[str] = set()

    def visit(relative: str) -> None:
        path = PurePosixPath(relative)
        if (path.is_absolute() or path.suffix != '.py'
                or any(part in {'', '.', '..'} for part in path.parts)):
            raise NumericSourceError('numeric config dependency path is unsafe')
        normalized = path.as_posix()
        if normalized in active:
            raise NumericSourceError('numeric config dependency cycle is invalid')
        if normalized in visited:
            return
        current = resolve_numeric_file(
            root, Path(normalized), 'numeric config dependency')
        blob = _blob(root, git_commit, normalized)
        if current.read_bytes() != blob:
            raise NumericSourceError(
                f'numeric config dependency differs from clean commit: '
                f'{normalized}')
        try:
            tree = ast.parse(blob, filename=normalized)
        except (SyntaxError, ValueError) as error:
            raise NumericSourceError(
                f'numeric config dependency is invalid Python: {normalized}') \
                from error
        bases: object = ()
        for node in tree.body:
            if (isinstance(node, (ast.Assign, ast.AnnAssign))
                    and ((isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name)
                                  and target.id == '_base_'
                                  for target in node.targets))
                         or (isinstance(node, ast.AnnAssign)
                             and isinstance(node.target, ast.Name)
                             and node.target.id == '_base_'))):
                try:
                    bases = ast.literal_eval(node.value)
                except (TypeError, ValueError) as error:
                    raise NumericSourceError(
                        f'numeric config _base_ is not literal: {normalized}') \
                        from error
                break
        if isinstance(bases, str):
            bases = (bases,)
        if (not isinstance(bases, (list, tuple))
                or any(not isinstance(base, str) or not base for base in bases)):
            raise NumericSourceError(
                f'numeric config _base_ is invalid: {normalized}')
        active.add(normalized)
        for base in bases:
            base_path = PurePosixPath(base)
            if base_path.is_absolute():
                raise NumericSourceError(
                    f'numeric config base is absolute: {normalized}')
            combined = path.parent.joinpath(base_path)
            parts: list[str] = []
            for part in combined.parts:
                if part in {'', '.'}:
                    continue
                if part == '..':
                    if not parts:
                        raise NumericSourceError(
                            f'numeric config base escapes repository: '
                            f'{normalized}')
                    parts.pop()
                else:
                    parts.append(part)
            visit(PurePosixPath(*parts).as_posix())
        active.remove(normalized)
        visited[normalized] = hashlib.sha256(blob).hexdigest()

    visit(start)
    return tuple(
        {'path': path, 'sha256': visited[path]} for path in sorted(visited))


def build_numeric_source_binding(
        *, repository_root: Path, candidate: CandidateSpec,
        manifest_path: Path, policy_path: Path,
        authority_path: Path = Path('optimization/coco_train2017_authority.json'),
        git_commit: str | None = None) -> dict[str, str]:
    root = Path(repository_root).resolve()
    commit = git_commit or clean_git_commit(root)
    if not _COMMIT.fullmatch(commit):
        raise NumericSourceError('numeric source commit is invalid')
    relative = {
        'manifest': _relative(root, manifest_path, 'manifest'),
        'config': _relative(root, candidate.config, 'config'),
        'checkpoint': _relative(root, candidate.checkpoint, 'checkpoint'),
        'policy': _relative(root, policy_path, 'policy'),
        'authority': _relative(root, authority_path, 'train authority'),
    }
    blobs = {
        name: _blob(root, commit, relative[name])
        for name in ('manifest', 'config', 'policy', 'authority')}
    validate_numeric_config_closure(
        root, Path(relative['config']), git_commit=commit)
    if relative['policy'] != relative['config']:
        validate_numeric_config_closure(
            root, Path(relative['policy']), git_commit=commit)
    for name, blob in blobs.items():
        current = root / relative[name]
        if not current.is_file() or current.read_bytes() != blob:
            raise NumericSourceError(
                f'numeric {name} differs from clean commit')
    selected, row_sha = _candidate_row(blobs['manifest'], candidate.id)
    if selected != candidate:
        raise NumericSourceError('numeric candidate differs from manifest row')
    checkpoint = root / relative['checkpoint']
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != candidate.checkpoint_sha256:
        raise NumericSourceError('numeric checkpoint differs from manifest sha256')
    return {
        'git_commit': commit,
        'manifest_path': relative['manifest'],
        'manifest_sha256': hashlib.sha256(blobs['manifest']).hexdigest(),
        'candidate_id': candidate.id,
        'candidate_row_sha256': row_sha,
        'config_path': relative['config'],
        'config_sha256': hashlib.sha256(blobs['config']).hexdigest(),
        'checkpoint_path': relative['checkpoint'],
        'checkpoint_sha256': checkpoint_sha,
        'policy_path': relative['policy'],
        'policy_sha256': hashlib.sha256(blobs['policy']).hexdigest(),
        'authority_path': relative['authority'],
        'authority_sha256': hashlib.sha256(blobs['authority']).hexdigest(),
    }


def validate_numeric_source_binding(
        value: Mapping[str, Any], *, repository_root: Path,
        candidate: CandidateSpec, manifest_path: Path) -> dict[str, str]:
    if (not isinstance(value, Mapping) or set(value) != _FIELDS
            or any(not isinstance(item, str) or not item
                   for item in value.values())):
        raise NumericSourceError('numeric source fields are invalid')
    normalized = dict(value)
    if (not _COMMIT.fullmatch(normalized['git_commit'])
            or any(not _SHA.fullmatch(normalized[name]) for name in (
                'manifest_sha256', 'candidate_row_sha256', 'config_sha256',
                'checkpoint_sha256', 'policy_sha256', 'authority_sha256'))):
        raise NumericSourceError('numeric source hashes are invalid')
    expected = build_numeric_source_binding(
        repository_root=repository_root, candidate=candidate,
        manifest_path=manifest_path,
        policy_path=Path(normalized['policy_path']),
        authority_path=Path(normalized['authority_path']),
        git_commit=normalized['git_commit'])
    if normalized != expected:
        raise NumericSourceError('numeric source binding disagrees with inputs')
    return normalized


__all__ = [
    'NumericSourceError', 'build_numeric_source_binding', 'file_sha256',
    'resolve_numeric_file', 'validate_numeric_config_closure',
    'validate_numeric_source_binding']
