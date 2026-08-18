"""Strict immutable campaign manifest parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


class ManifestError(ValueError):
    """The campaign manifest is unsafe or does not match its schema."""


def _require_fields(
        value: Mapping[str, Any], allowed: set[str], required: set[str],
        context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ManifestError(
            f'{context} has unknown fields: {sorted(unknown)}')
    missing = required - set(value)
    if missing:
        raise ManifestError(
            f'{context} is missing fields: {sorted(missing)}')


def _relative_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f'{context} must be a non-empty relative path')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ManifestError(
            f'{context} must be relative and contain no parent traversal')
    return path


@dataclass(frozen=True)
class RunSpec:
    id: str
    kind: str
    config: Path
    work_dir: Path
    depends_on: tuple[str, ...]
    artifacts: tuple[str, ...]
    max_attempts: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'RunSpec':
        allowed = {
            'id', 'kind', 'config', 'work_dir', 'depends_on', 'artifacts',
            'max_attempts'
        }
        _require_fields(
            value, allowed,
            {'id', 'kind', 'config', 'work_dir', 'depends_on', 'artifacts',
             'max_attempts'},
            'run')
        run_id = value['id']
        if not isinstance(run_id, str) or not run_id or '/' in run_id:
            raise ManifestError('run id must be a non-empty filesystem-safe string')
        if value['kind'] not in {'train', 'export'}:
            raise ManifestError(f'run {run_id} has invalid kind')
        dependencies = value['depends_on']
        artifacts = value['artifacts']
        if (not isinstance(dependencies, list)
                or not all(isinstance(item, str) for item in dependencies)):
            raise ManifestError(f'run {run_id} depends_on must be strings')
        admitted_artifacts = {'checkpoint', 'metrics', 'submission'}
        if (not isinstance(artifacts, list) or not artifacts
                or not all(item in admitted_artifacts for item in artifacts)):
            raise ManifestError(f'run {run_id} has undeclared artifact validator')
        attempts = value['max_attempts']
        if not isinstance(attempts, int) or not 1 <= attempts <= 10:
            raise ManifestError(f'run {run_id} has invalid max_attempts')
        return cls(
            id=run_id,
            kind=value['kind'],
            config=_relative_path(value['config'], f'run {run_id} config'),
            work_dir=_relative_path(
                value['work_dir'], f'run {run_id} work_dir'),
            depends_on=tuple(dependencies),
            artifacts=tuple(artifacts),
            max_attempts=attempts)


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    reproduction_id: str
    paper: Mapping[str, str]
    runs: tuple[RunSpec, ...]


def load_manifest(path: Path | str) -> Manifest:
    """Load and validate an immutable reproduction manifest."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f'cannot load manifest {manifest_path}: {error}') from error
    if not isinstance(raw, dict):
        raise ManifestError('manifest root must be an object')
    _require_fields(
        raw, {'schema_version', 'reproduction_id', 'paper', 'runs'},
        {'schema_version', 'reproduction_id', 'paper', 'runs'}, 'manifest')
    if raw['schema_version'] != 1:
        raise ManifestError('unsupported manifest schema_version')
    if not isinstance(raw['reproduction_id'], str) or not raw['reproduction_id']:
        raise ManifestError('reproduction_id must be a non-empty string')
    paper = raw['paper']
    if not isinstance(paper, dict):
        raise ManifestError('paper must be an object')
    _require_fields(
        paper, {'path', 'sha256', 'title'}, {'path', 'sha256', 'title'},
        'paper')
    _relative_path(paper['path'], 'paper path')
    if (not isinstance(paper['sha256'], str)
            or len(paper['sha256']) != 64):
        raise ManifestError('paper sha256 must have 64 hexadecimal characters')
    if not isinstance(raw['runs'], list) or not raw['runs']:
        raise ManifestError('runs must be a non-empty list')
    runs = tuple(RunSpec.from_dict(item) for item in raw['runs'])
    run_ids = [run.id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ManifestError('duplicate run id in manifest')
    known_ids = set(run_ids)
    for run in runs:
        unknown = set(run.depends_on) - known_ids
        if unknown:
            raise ManifestError(
                f'run {run.id} has unknown dependencies: {sorted(unknown)}')
        if run.id in run.depends_on:
            raise ManifestError(f'run {run.id} depends on itself')
    return Manifest(
        schema_version=raw['schema_version'],
        reproduction_id=raw['reproduction_id'],
        paper=dict(paper),
        runs=runs)

