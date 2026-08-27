"""Strict, reproducible candidate-manifest validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None

_CANDIDATE_FIELDS = frozenset({
    'id', 'route', 'kind', 'config', 'checkpoint', 'checkpoint_sha256',
    'seed', 'features',
})
_ROUTES = frozenset({
    'baseline', 'accuracy-first', 'structural-pif', 'ssm-quant-pwl',
})
_KINDS = frozenset({
    'float', 'structural', 'fake-quant', 'pwl', 'binary-qk', 'integrated',
})
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_IDENTIFIER = re.compile(r'^[a-z0-9][a-z0-9._-]{0,127}$')


class CandidateManifestError(ValueError):
    """Raised when a candidate manifest is not a stable, safe contract."""


@dataclass(frozen=True)
class CandidateSpec:
    id: str
    route: Literal['baseline', 'accuracy-first', 'structural-pif', 'ssm-quant-pwl']
    kind: Literal['float', 'structural', 'fake-quant', 'pwl', 'binary-qk', 'integrated']
    config: Path
    checkpoint: Path
    checkpoint_sha256: str
    seed: int
    features: Mapping[str, JSONScalar]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateSpec:
        if not isinstance(value, Mapping):
            raise CandidateManifestError('candidate must be an object')
        unknown = set(value) - _CANDIDATE_FIELDS
        missing = _CANDIDATE_FIELDS - set(value)
        if unknown:
            raise CandidateManifestError(f'candidate has unknown fields: {sorted(unknown)}')
        if missing:
            raise CandidateManifestError(f'candidate is missing fields: {sorted(missing)}')

        identifier = value['id']
        if (
                not isinstance(identifier, str)
                or not _IDENTIFIER.fullmatch(identifier)):
            raise CandidateManifestError(
                'candidate id must be a path-safe lowercase identifier')

        route = value['route']
        if not isinstance(route, str) or route not in _ROUTES:
            raise CandidateManifestError(f'invalid route: {route!r}')
        kind = value['kind']
        if not isinstance(kind, str) or kind not in _KINDS:
            raise CandidateManifestError(f'invalid kind: {kind!r}')

        config = _safe_relative_path(value['config'], 'config')
        checkpoint = _safe_relative_path(value['checkpoint'], 'checkpoint')

        checksum = value['checkpoint_sha256']
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise CandidateManifestError('checkpoint_sha256 must be a lowercase sha256')

        seed = value['seed']
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CandidateManifestError('seed must be an integer')

        features = value['features']
        if not isinstance(features, Mapping) or not all(
                isinstance(key, str) and _is_json_scalar(item)
                for key, item in features.items()):
            raise CandidateManifestError('features must map strings to JSON scalar values')

        return cls(
            id=identifier,
            route=route,
            kind=kind,
            config=config,
            checkpoint=checkpoint,
            checkpoint_sha256=checksum,
            seed=seed,
            features=MappingProxyType(dict(features)),
        )


def load_candidate_manifest(path: Path | str) -> tuple[CandidateSpec, ...]:
    """Load a version-one manifest without accepting untracked semantics."""
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateManifestError(f'cannot load candidate manifest: {error}') from error

    return parse_candidate_manifest(value)


def parse_candidate_manifest(value: object) -> tuple[CandidateSpec, ...]:
    """Validate a decoded manifest, including a blob loaded from Git."""
    if not isinstance(value, Mapping):
        raise CandidateManifestError('manifest must be an object')
    if set(value) != {'schema_version', 'candidates'}:
        raise CandidateManifestError('manifest has unknown fields or is missing required fields')
    if value['schema_version'] != 1:
        raise CandidateManifestError('schema_version must be 1')
    candidates = value['candidates']
    if not isinstance(candidates, list):
        raise CandidateManifestError('candidates must be a list')

    parsed = tuple(CandidateSpec.from_dict(candidate) for candidate in candidates)
    ids = [candidate.id for candidate in parsed]
    if len(ids) != len(set(ids)):
        raise CandidateManifestError('candidate ids must be unique; duplicate id found')
    return parsed


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CandidateManifestError(f'{field} must be a non-empty relative path')
    path = Path(value)
    if path.is_absolute():
        raise CandidateManifestError(f'{field} must be a relative path')
    if any(part in {'.', '..'} for part in path.parts):
        raise CandidateManifestError(f'{field} path traversal is not allowed')
    if any(character in value for character in ('\x00', '\n', '\r', ';', '|', '&', '`', '$')):
        raise CandidateManifestError(f'{field} must not contain shell commands')
    return path


def _is_json_scalar(value: Any) -> bool:
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )
