"""Manifest-authorized, data-only candidate checkpoint loading."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .artifacts import lexical_repository_root
from .evaluation import (
    build_source_binding, resolve_project_asset_root,
    resolve_shared_asset_exposure)
from .schema import CandidateSpec, load_candidate_manifest
from .source import clean_git_commit


@dataclass(frozen=True)
class AuthorizedCandidate:
    candidate: CandidateSpec
    config_path: Path
    checkpoint_path: Path
    source: Mapping[str, str]


_APPROVED_CHECKPOINT_ROOTS = (Path('work_dirs/reproduction'),)


@dataclass(frozen=True)
class _ConfigSnapshot:
    candidate: CandidateSpec
    path: Path
    sha256: str
    config: Any
    reference: Mapping[str, Any]

    def identity(self) -> tuple[CandidateSpec, Path, str, str]:
        return (
            self.candidate, self.path, self.sha256,
            json.dumps(
                self.reference, sort_keys=True, separators=(',', ':'),
                allow_nan=False))


class ConfigAuthority:
    """A public locator whose Config is reconstructed on every consumption."""

    __slots__ = (
        '_candidate_id', '_manifest_path', '_reference', '_repository_root')

    def __init__(self, *args, **kwargs):
        raise TypeError('ConfigAuthority must be created by an authorizer')

    @classmethod
    def _create(
            cls, *, repository_root: Path, manifest_path: Path,
            candidate_id: str, reference: Mapping[str, Any]):
        value = object.__new__(cls)
        object.__setattr__(
            value, '_repository_root',
            Path(repository_root).resolve(strict=True))
        object.__setattr__(
            value, '_manifest_path', Path(manifest_path).resolve(strict=True))
        object.__setattr__(value, '_candidate_id', candidate_id)
        object.__setattr__(
            value, '_reference', copy.deepcopy(dict(reference)))
        return value

    def _snapshot(self) -> _ConfigSnapshot:
        return _reconstruct_config_snapshot(
            self._repository_root, self._manifest_path,
            self._candidate_id, self._reference)

    def _require_exact(self, expected: 'ConfigAuthority') -> None:
        if not isinstance(expected, ConfigAuthority):
            raise TypeError('expected ConfigAuthority is invalid')
        actual_identity = self._locator_identity()
        expected_identity = expected._locator_identity()
        if actual_identity != expected_identity:
            raise ValueError(
                'ConfigAuthority locator differs from the expected stage source')

    def _locator_identity(self) -> tuple[Path, Path, str, str]:
        return (
            Path(self._repository_root), Path(self._manifest_path),
            str(self._candidate_id),
            json.dumps(
                self._reference, sort_keys=True, separators=(',', ':'),
                allow_nan=False))

    @property
    def path(self) -> Path:
        return self._snapshot().path

    @property
    def sha256(self) -> str:
        return self._snapshot().sha256

    @property
    def candidate(self) -> CandidateSpec:
        return self._snapshot().candidate

    @staticmethod
    def _config_fingerprint(config: Any) -> str:
        try:
            serialized = config.dump()
        except (AttributeError, OSError, TypeError, ValueError) as error:
            raise ValueError('ConfigAuthority Config is not serializable') from error
        if not isinstance(serialized, str):
            raise ValueError('ConfigAuthority Config is not serializable')
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def verify(self) -> None:
        first = self._snapshot()
        second = self._snapshot()
        if (
                first.identity() != second.identity()
                or self._config_fingerprint(first.config) !=
                self._config_fingerprint(second.config)):
            raise ValueError('ConfigAuthority source changed during verification')

    def load_config(self) -> Any:
        first = self._snapshot()
        value = copy.deepcopy(first.config)
        second = self._snapshot()
        if (
                first.identity() != second.identity()
                or self._config_fingerprint(first.config) !=
                self._config_fingerprint(second.config)):
            raise ValueError('ConfigAuthority source changed during parsing')
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _lexical_file(root: Path, relative: Path, *, label: str) -> Path:
    """Reject a symlink in any caller-visible path component before resolve."""
    if relative.is_absolute() or any(part in {'.', '..'} for part in relative.parts):
        raise ValueError(f'{label} must be a safe relative path')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'{label} path must not use symlinks')
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError(f'{label} must be an existing contained file') from error
    if not resolved.is_file():
        raise ValueError(f'{label} must be a regular file')
    return resolved


def _approved_checkpoint_root(relative: Path) -> Path:
    for approved in _APPROVED_CHECKPOINT_ROOTS:
        if relative.parts[:len(approved.parts)] == approved.parts:
            return approved
    raise ValueError(
        'candidate checkpoint is not under an approved shared asset root')


def authorize_manifest_candidate(
        repository_root: Path, manifest_path: Path, candidate_id: str,
        ) -> AuthorizedCandidate:
    """Select and bind exactly one tracked candidate before checkpoint load."""
    root = lexical_repository_root(repository_root)
    manifest_relative = Path(manifest_path)
    if manifest_relative.is_absolute():
        try:
            manifest_relative = manifest_relative.relative_to(root)
        except ValueError as error:
            raise ValueError('candidate manifest must be repository-contained') from error
    manifest = _lexical_file(root, manifest_relative, label='candidate manifest')
    matches = [
        item for item in load_candidate_manifest(manifest)
        if item.id == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f'manifest must contain exactly one candidate {candidate_id!r}')
    candidate = matches[0]
    commit = clean_git_commit(root)
    source = build_source_binding(
        repository_root=root, candidate=candidate,
        manifest_path=manifest, git_commit=commit)
    config = _lexical_file(root, candidate.config, label='candidate config')
    asset_root = resolve_project_asset_root(root)
    checkpoint_root = _approved_checkpoint_root(candidate.checkpoint)
    resolve_shared_asset_exposure(
        root, asset_root, checkpoint_root, label='candidate checkpoint')
    checkpoint = _lexical_file(
        asset_root, candidate.checkpoint, label='candidate checkpoint')
    actual = _sha256(checkpoint)
    if actual != candidate.checkpoint_sha256:
        raise ValueError(
            f'candidate checkpoint sha256 mismatch: expected '
            f'{candidate.checkpoint_sha256}, got {actual}')
    return AuthorizedCandidate(candidate, config, checkpoint, source)


def authorize_candidate_checkpoint_reference(
        repository_root: Path, manifest_path: Path, candidate: CandidateSpec,
        reference: object) -> Path:
    """Bind one serialized logical checkpoint reference to its approved file."""
    if not isinstance(reference, Mapping) or set(reference) != {
            'path', 'sha256'}:
        raise ValueError('candidate checkpoint reference is invalid')
    expected = {
        'path': candidate.checkpoint.as_posix(),
        'sha256': candidate.checkpoint_sha256,
    }
    if dict(reference) != expected:
        raise ValueError(
            'candidate checkpoint reference disagrees with manifest')
    try:
        authorized = authorize_manifest_candidate(
            repository_root, manifest_path, candidate.id)
    except (
            OSError, RuntimeError, subprocess.SubprocessError,
            ValueError) as error:
        raise ValueError(
            f'candidate checkpoint is not authorized: {error}') from error
    if authorized.candidate != candidate:
        raise ValueError(
            'candidate checkpoint reference differs from authorized manifest')
    return authorized.checkpoint_path


def authorized_tracked_file(
        repository_root: Path, relative: Path | str, *, commit: str,
        label: str) -> Path:
    """Resolve a nonsymlink source file and require its exact Git blob."""
    root = lexical_repository_root(repository_root)
    path = _lexical_file(root, Path(relative), label=label)
    relative_text = path.relative_to(root).as_posix()
    try:
        blob = subprocess.run(
            ['git', 'show', f'{commit}:{relative_text}'], cwd=root,
            check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f'{label} is not tracked at the authorized commit') from error
    if path.read_bytes() != blob:
        raise ValueError(f'{label} differs from the authorized Git blob')
    return path


def _manifest_absolute(root: Path, manifest_path: Path) -> Path:
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    return manifest.resolve(strict=True)


def _tracked_blob(root: Path, commit: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            ['git', 'show', f'{commit}:{relative}'], cwd=root,
            check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f'authenticated Config dependency is not tracked: {relative}') \
            from error


def _safe_closure_relative(value: str) -> Path:
    relative = Path(value)
    if (
            relative.is_absolute() or relative.suffix != '.py'
            or not relative.parts
            or any(part in {'', '.', '..'} for part in relative.parts)):
        raise ValueError('authenticated Config closure path is unsafe')
    return relative


def _write_private_config_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    path.chmod(0o400)


def _private_closure_hashes(
        private_root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, checksum in expected.items():
        relative = _safe_closure_relative(name)
        path = private_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('authenticated Config closure file changed')
        actual = _sha256(path)
        if actual != checksum:
            raise ValueError('authenticated Config closure hash changed')
        observed[relative.as_posix()] = actual
    extra = tuple(
        path for path in private_root.rglob('*')
        if path.is_file()
        and path.relative_to(private_root).as_posix() not in expected)
    if extra:
        raise ValueError('authenticated Config parser created extra files')
    return observed


def _parse_authenticated_tracked_config(
        root: Path, *, start: Path, commit: str,
        closure: tuple[dict[str, str], ...]):
    """Parse only a private exact-blob closure, never a live source path."""
    from mmengine.config import Config

    expected = {
        _safe_closure_relative(item['path']).as_posix(): item['sha256']
        for item in closure
    }
    start_relative = _safe_closure_relative(start.as_posix())
    if start_relative.as_posix() not in expected:
        raise ValueError('authenticated Config root is missing from closure')
    blobs: dict[str, bytes] = {}
    for relative, checksum in expected.items():
        payload = _tracked_blob(root, commit, relative)
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError('authenticated Config Git blob hash changed')
        if (root / relative).read_bytes() != payload:
            raise ValueError('authenticated Config live dependency changed')
        blobs[relative] = payload

    with tempfile.TemporaryDirectory(prefix='mambapose-config-') as name:
        private_root = Path(name)
        if private_root.is_symlink() or not private_root.is_dir():
            raise ValueError('authenticated Config temporary root is invalid')
        private_root.chmod(0o700)
        directories = {private_root}
        for relative in sorted(blobs):
            destination = private_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            cursor = destination.parent
            while cursor != private_root.parent:
                directories.add(cursor)
                if cursor == private_root:
                    break
                cursor = cursor.parent
            _write_private_config_file(destination, blobs[relative])
        before = _private_closure_hashes(private_root, expected)
        for directory in directories:
            directory.chmod(0o500)
        try:
            config = Config.fromfile(private_root / start_relative)
            after = _private_closure_hashes(private_root, expected)
        finally:
            for directory in directories:
                directory.chmod(0o700)
        if before != after or after != expected:
            raise ValueError('authenticated Config closure changed during parse')

    # Force serialization after the private tree is gone. This rejects a
    # parser result that retained lazy path-backed state.
    ConfigAuthority._config_fingerprint(config)
    return config


def _tracked_config_snapshot(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec) -> _ConfigSnapshot:
    """Rebuild a tracked Config and its full public source evidence."""
    from .numeric_source import validate_numeric_config_closure

    root = lexical_repository_root(repository_root)
    candidate_id = candidate if isinstance(candidate, str) else candidate.id
    authorized = authorize_manifest_candidate(root, manifest_path, candidate_id)
    if isinstance(candidate, CandidateSpec) and authorized.candidate != candidate:
        raise ValueError('candidate differs from the authorized manifest entry')
    commit = authorized.source['git_commit']

    def source_snapshot() -> tuple[
            Path, bytes, tuple[dict[str, str], ...]]:
        path = authorized_tracked_file(
            root, authorized.candidate.config, commit=commit,
            label='candidate config')
        closure = validate_numeric_config_closure(
            root, authorized.candidate.config, git_commit=commit)
        return path, path.read_bytes(), closure

    path, payload, closure = source_snapshot()
    config = _parse_authenticated_tracked_config(
        root, start=authorized.candidate.config, commit=commit,
        closure=closure)
    after_authorized = authorize_manifest_candidate(
        root, manifest_path, candidate_id)
    after_path, after_payload, after_closure = source_snapshot()
    if (
            after_authorized != authorized
            or after_authorized.source != authorized.source
            or after_authorized.source.get('git_commit') != commit
            or after_path != path or after_payload != payload
            or after_closure != closure):
        raise ValueError('candidate config authority changed during parsing')
    checksum = hashlib.sha256(payload).hexdigest()
    return _ConfigSnapshot(
        candidate=authorized.candidate, path=path, sha256=checksum,
        config=config,
        reference={
            'kind': 'tracked-candidate-config-v1',
            'config_path': authorized.candidate.config.as_posix(),
            'config_sha256': checksum,
            'git_commit': commit,
            'closure': [dict(item) for item in closure],
        })


def authorize_tracked_config(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec) -> ConfigAuthority:
    """Return a locator for a tracked Config, never the parsed Config itself."""
    root = lexical_repository_root(repository_root)
    snapshot = _tracked_config_snapshot(root, manifest_path, candidate)
    authority = ConfigAuthority._create(
        repository_root=root,
        manifest_path=_manifest_absolute(root, manifest_path),
        candidate_id=snapshot.candidate.id, reference=snapshot.reference)
    authority.verify()
    return authority


def _repository_relative(root: Path, path: Path, *, label: str) -> Path:
    supplied = Path(path)
    lexical = supplied if supplied.is_absolute() else root / supplied
    lexical = lexical.absolute()
    try:
        relative = lexical.relative_to(root.absolute())
    except ValueError as error:
        raise ValueError(f'{label} escapes repository') from error
    if any(part in {'', '.', '..'} for part in relative.parts):
        raise ValueError(f'{label} path is unsafe')
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not use symlinks')
    return relative


def _config_from_bytes(payload: bytes, *, label: str):
    from mmengine.config import Config

    try:
        return Config.fromstring(payload.decode('utf-8'), '.py')
    except (UnicodeDecodeError, OSError, TypeError, ValueError) as error:
        raise ValueError(f'{label} is not a valid materialized Config') from error


def parse_authenticated_config_bytes(
        payload: bytes, *, expected_sha256: str, label: str):
    """Parse a captured standalone Config only after authenticating its bytes."""
    if not isinstance(payload, bytes):
        raise TypeError(f'{label} payload must be bytes')
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f'{label} captured bytes hash mismatch')
    config = _config_from_bytes(payload, label=label)
    ConfigAuthority._config_fingerprint(config)
    return config


def _pwl_runtime_config_snapshot(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec, *,
        conversion_path: Path) -> _ConfigSnapshot:
    """Rebuild a runtime Config from a public-valid convert record."""
    from .numeric_runtime import validate_numeric_convert_artifact

    root = lexical_repository_root(repository_root)
    candidate_id = candidate if isinstance(candidate, str) else candidate.id
    authorized = authorize_manifest_candidate(root, manifest_path, candidate_id)
    if isinstance(candidate, CandidateSpec) and authorized.candidate != candidate:
        raise ValueError('candidate differs from the authorized manifest entry')
    if authorized.candidate.features.get('numeric_kind') != 'pwl':
        raise ValueError('runtime ConfigAuthority requires a PWL candidate')
    conversion_relative = _repository_relative(
        root, conversion_path, label='PWL conversion artifact')
    conversion_file = _lexical_file(
        root, conversion_relative, label='PWL conversion artifact')

    def source_snapshot():
        try:
            conversion_payload = conversion_file.read_bytes()
            conversion = json.loads(conversion_payload)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError('PWL conversion artifact is invalid') from error
        validate_numeric_convert_artifact(
            conversion, candidate=authorized.candidate,
            repository_root=root, manifest_path=manifest_path,
            artifact_path=conversion_file)
        reference = conversion['result']['runtime_config']
        runtime_relative = _repository_relative(
            root, Path(reference['path']), label='PWL runtime config')
        runtime_path = _lexical_file(
            root, runtime_relative, label='PWL runtime config')
        runtime_payload = runtime_path.read_bytes()
        runtime_sha256 = hashlib.sha256(runtime_payload).hexdigest()
        if runtime_sha256 != reference['sha256']:
            raise ValueError('PWL runtime config hash changed')
        return (
            conversion_payload, runtime_path, runtime_payload,
            runtime_sha256)

    conversion_payload, path, payload, checksum = source_snapshot()
    config = parse_authenticated_config_bytes(
        payload, expected_sha256=checksum, label='PWL runtime config')
    after = source_snapshot()
    if after != (conversion_payload, path, payload, checksum):
        raise ValueError('PWL runtime config authority changed during parsing')
    return _ConfigSnapshot(
        candidate=authorized.candidate, path=path, sha256=checksum,
        config=config,
        reference={
            'kind': 'pwl-convert-runtime-v1',
            'conversion_path': conversion_relative.as_posix(),
            'conversion_sha256': hashlib.sha256(
                conversion_payload).hexdigest(),
            'runtime_path': path.relative_to(root).as_posix(),
            'runtime_sha256': checksum,
            'source_git_commit': authorized.source['git_commit'],
            'source_manifest_sha256': authorized.source['manifest_sha256'],
            'source_config_sha256': authorized.source['config_sha256'],
        })


def authorize_pwl_runtime_config(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec, *,
        conversion_path: Path) -> ConfigAuthority:
    """Return a locator for the exact public-valid PWL runtime Config."""
    root = lexical_repository_root(repository_root)
    snapshot = _pwl_runtime_config_snapshot(
        root, manifest_path, candidate, conversion_path=conversion_path)
    authority = ConfigAuthority._create(
        repository_root=root,
        manifest_path=_manifest_absolute(root, manifest_path),
        candidate_id=snapshot.candidate.id, reference=snapshot.reference)
    authority.verify()
    return authority


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluation_config(authority: ConfigAuthority, flip_test: bool):
    from .evaluation import build_deterministic_evaluation_config

    return build_deterministic_evaluation_config(
        authority.candidate, flip_test, config=authority.load_config())


def materialize_evaluation_config_authority(
        authority: ConfigAuthority, *, flip_test: bool,
        config_path: Path, authority_path: Path) -> ConfigAuthority:
    """Write the exact transformed config and a child-consumed authority."""
    if not isinstance(authority, ConfigAuthority):
        raise TypeError('base authority must be a ConfigAuthority locator')
    base_snapshot = authority._snapshot()
    base_reference = dict(base_snapshot.reference)
    if base_reference.get('kind') != 'pwl-convert-runtime-v1':
        raise ValueError('evaluation materialization requires PWL runtime authority')
    root = Path(authority._repository_root)
    config_relative = _repository_relative(
        root, config_path, label='materialized evaluation config')
    authority_relative = _repository_relative(
        root, authority_path, label='materialized config authority')
    if (config_relative.parts[:2] != ('work_dirs', 'optimization')
            or authority_relative.parts[:2] != ('work_dirs', 'optimization')):
        raise ValueError('materialized evaluation authority is not canonical')
    config = _evaluation_config(authority, flip_test)
    serialized = config.dump()
    if not isinstance(serialized, str):
        raise ValueError('evaluation Config is not serializable')
    payload = serialized.encode('utf-8')
    checksum = hashlib.sha256(payload).hexdigest()
    config_file = root / config_relative
    authority_file = root / authority_relative
    _write_atomic(config_file, payload)
    record = {
        'schema_version': 1,
        'candidate_id': base_snapshot.candidate.id,
        'base': {
            name: base_reference[name] for name in (
                'kind', 'conversion_path', 'conversion_sha256')},
        'transform': {
            'kind': 'deterministic-coco-evaluation-v1',
            'flip_test': flip_test,
            'seed': base_snapshot.candidate.seed,
        },
        'materialized': {
            'path': config_relative.as_posix(),
            'sha256': checksum,
        },
    }
    _write_atomic(
        authority_file,
        (json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
         + '\n').encode('utf-8'))
    return load_materialized_config_authority(
        root, authority._manifest_path, base_snapshot.candidate, authority_file)


def _materialized_config_snapshot(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec,
        authority_path: Path) -> _ConfigSnapshot:
    """Rebuild an evaluation Config from its public materialized authority."""
    root = lexical_repository_root(repository_root)
    candidate_id = candidate if isinstance(candidate, str) else candidate.id
    authorized = authorize_manifest_candidate(root, manifest_path, candidate_id)
    if isinstance(candidate, CandidateSpec) and authorized.candidate != candidate:
        raise ValueError('candidate differs from the authorized manifest entry')
    authority_relative = _repository_relative(
        root, authority_path, label='materialized config authority')
    authority_file = _lexical_file(
        root, authority_relative, label='materialized config authority')

    def snapshot():
        authority_payload = authority_file.read_bytes()
        try:
            record = json.loads(authority_payload)
        except json.JSONDecodeError as error:
            raise ValueError('materialized config authority is invalid JSON') \
                from error
        if (
                not isinstance(record, Mapping)
                or set(record) != {
                    'schema_version', 'candidate_id', 'base',
                    'transform', 'materialized'}
                or record.get('schema_version') != 1
                or record.get('candidate_id') != authorized.candidate.id
                or record.get('transform') != {
                    'kind': 'deterministic-coco-evaluation-v1',
                    'flip_test': record.get('transform', {}).get('flip_test'),
                    'seed': authorized.candidate.seed}
                or not isinstance(record['transform']['flip_test'], bool)
                or not isinstance(record.get('base'), Mapping)
                or set(record['base']) != {
                    'kind', 'conversion_path', 'conversion_sha256'}
                or record['base'].get('kind') != 'pwl-convert-runtime-v1'
                or not isinstance(record.get('materialized'), Mapping)
                or set(record['materialized']) != {'path', 'sha256'}):
            raise ValueError('materialized config authority fields are invalid')
        conversion_relative = _repository_relative(
            root, Path(record['base']['conversion_path']),
            label='PWL conversion artifact')
        conversion_file = _lexical_file(
            root, conversion_relative, label='PWL conversion artifact')
        if _sha256(conversion_file) != record['base']['conversion_sha256']:
            raise ValueError('materialized config base conversion changed')
        base = _pwl_runtime_config_snapshot(
            root, manifest_path, authorized.candidate,
            conversion_path=conversion_file)
        expected_config = _evaluation_config(
            _authority_from_snapshot(
                root, manifest_path, base),
            record['transform']['flip_test'])
        serialized = expected_config.dump()
        if not isinstance(serialized, str):
            raise ValueError('materialized evaluation Config is not serializable')
        expected_payload = serialized.encode('utf-8')
        expected_sha = hashlib.sha256(expected_payload).hexdigest()
        materialized_relative = _repository_relative(
            root, Path(record['materialized']['path']),
            label='materialized evaluation config')
        materialized_file = _lexical_file(
            root, materialized_relative, label='materialized evaluation config')
        actual_payload = materialized_file.read_bytes()
        if (record['materialized']['sha256'] != expected_sha
                or actual_payload != expected_payload):
            raise ValueError('materialized evaluation config changed')
        return (
            authority_payload, materialized_file, actual_payload,
            expected_sha, expected_config, record)

    initial = snapshot()
    authority_payload, path, payload, checksum, _expected, record = initial
    config = parse_authenticated_config_bytes(
        payload, expected_sha256=checksum,
        label='materialized evaluation config')
    after = snapshot()
    if after[:4] != initial[:4] or after[5] != record:
        raise ValueError('materialized config authority changed during parsing')

    return _ConfigSnapshot(
        candidate=authorized.candidate, path=path, sha256=checksum,
        config=config,
        reference={
            'kind': 'materialized-evaluation-v1',
            'authority_path': authority_relative.as_posix(),
            'authority_sha256': hashlib.sha256(
                authority_payload).hexdigest(),
            'base_conversion_path': record['base']['conversion_path'],
            'base_conversion_sha256': record['base']['conversion_sha256'],
            'materialized_path': path.relative_to(root).as_posix(),
            'materialized_sha256': checksum,
        })


def _authority_from_snapshot(
        root: Path, manifest_path: Path,
        snapshot: _ConfigSnapshot) -> ConfigAuthority:
    return ConfigAuthority._create(
        repository_root=root,
        manifest_path=_manifest_absolute(root, manifest_path),
        candidate_id=snapshot.candidate.id, reference=snapshot.reference)


def load_materialized_config_authority(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec,
        authority_path: Path) -> ConfigAuthority:
    """Return a reconstructable locator for one materialized evaluation Config."""
    root = lexical_repository_root(repository_root)
    snapshot = _materialized_config_snapshot(
        root, manifest_path, candidate, authority_path)
    authority = _authority_from_snapshot(root, manifest_path, snapshot)
    authority.verify()
    return authority


def _reconstruct_config_snapshot(
        repository_root: Path, manifest_path: Path, candidate_id: str,
        reference: Mapping[str, Any]) -> _ConfigSnapshot:
    """Rebuild one locator and reject every caller-selected substitution."""
    if not isinstance(reference, Mapping):
        raise ValueError('ConfigAuthority locator reference is invalid')
    kind = reference.get('kind')
    if kind == 'tracked-candidate-config-v1':
        snapshot = _tracked_config_snapshot(
            repository_root, manifest_path, candidate_id)
    elif kind == 'pwl-convert-runtime-v1':
        if not isinstance(reference.get('conversion_path'), str):
            raise ValueError('PWL runtime ConfigAuthority locator is invalid')
        snapshot = _pwl_runtime_config_snapshot(
            repository_root, manifest_path, candidate_id,
            conversion_path=Path(reference['conversion_path']))
    elif kind == 'materialized-evaluation-v1':
        if not isinstance(reference.get('authority_path'), str):
            raise ValueError(
                'materialized ConfigAuthority locator is invalid')
        snapshot = _materialized_config_snapshot(
            repository_root, manifest_path, candidate_id,
            Path(reference['authority_path']))
    else:
        raise ValueError('ConfigAuthority locator kind is invalid')
    if dict(snapshot.reference) != dict(reference):
        raise ValueError('ConfigAuthority locator evidence changed')
    return snapshot


def tensor_state(checkpoint: Path) -> dict[str, Tensor]:
    """Load the known MMPose mapping with PyTorch's restricted unpickler."""
    from mmengine.logging.history_buffer import HistoryBuffer

    safe: list[object] = [
        np.core.multiarray._reconstruct,
        np.core.multiarray.scalar,
        np.ndarray,
        np.dtype,
        HistoryBuffer,
        getattr,
    ]
    safe.extend(
        value for value in vars(np.dtypes).values()
        if isinstance(value, type))
    try:
        with torch.serialization.safe_globals(safe):
            payload = torch.load(
                checkpoint, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError(
            f'authorized checkpoint is not weights-only compatible: {error}') from error
    if isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict):
        payload = payload['state_dict']
    if (
            not isinstance(payload, dict)
            or not payload
            or not all(isinstance(key, str) and isinstance(value, Tensor)
                       for key, value in payload.items())):
        raise ValueError('checkpoint must contain a non-empty tensor state_dict')
    return dict(payload)


def _neutralize_initializers(value: Any) -> Any:
    """Return a copy with every implicit model initializer disabled."""
    if isinstance(value, Mapping):
        result = copy.deepcopy(value)
        for key in list(result):
            if key in {'pretrained', 'init_cfg'}:
                result[key] = None
            else:
                result[key] = _neutralize_initializers(result[key])
        return result
    if isinstance(value, list):
        return [_neutralize_initializers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_neutralize_initializers(item) for item in value)
    return copy.deepcopy(value)


def neutralize_model_initializers(config: Any) -> Any:
    """Copy a Config and prevent model construction from loading any asset."""
    result = copy.deepcopy(config)
    if not hasattr(result, 'model'):
        raise ValueError('config must define a model')
    result.model = _neutralize_initializers(result.model)
    return result


def load_tensor_state_strict(model: Any, state: Mapping[str, Tensor]) -> None:
    """Inject an exact, finite tensor state with no missing/extra coercions."""
    if (
            not isinstance(state, Mapping)
            or not state
            or not all(isinstance(key, str) and isinstance(value, Tensor)
                       for key, value in state.items())):
        raise ValueError('state must be a non-empty string-to-tensor mapping')
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing:
        raise ValueError(f'checkpoint has missing state keys: {missing}')
    if unexpected:
        raise ValueError(f'checkpoint has unexpected state keys: {unexpected}')
    for key, target in expected.items():
        value = state[key]
        if value.shape != target.shape:
            raise ValueError(
                f'checkpoint tensor shape mismatch for {key}: '
                f'{tuple(value.shape)} != {tuple(target.shape)}')
        if value.dtype != target.dtype:
            raise ValueError(
                f'checkpoint tensor dtype mismatch for {key}: '
                f'{value.dtype} != {target.dtype}')
        if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()):
            raise ValueError(f'checkpoint tensor is nonfinite for {key}')

    from mmengine.runner.checkpoint import _load_checkpoint_to_model

    _load_checkpoint_to_model(
        model, {'state_dict': dict(state)}, strict=True)
    loaded = model.state_dict()
    for key, value in state.items():
        actual = loaded[key]
        if (
                actual.shape != value.shape
                or actual.dtype != value.dtype
                or not torch.equal(actual.detach().cpu(), value.detach().cpu())):
            raise ValueError(
                f'checkpoint post-load verification failed for {key}')


def _build_authorized_model(
        authorized: AuthorizedCandidate, *, config_authority: ConfigAuthority,
        repository_root: Path, manifest_path: Path,
        device: str = 'cpu') -> Any:
    """Construct without implicit I/O, then inject authorized tensor state."""
    from mmpose.apis import init_model

    if not isinstance(config_authority, ConfigAuthority):
        raise TypeError('config_authority must be a ConfigAuthority locator')
    source_config = config_authority.load_config()
    safe_config = neutralize_model_initializers(source_config)
    model = init_model(safe_config, None, device=device)
    load_tensor_state_strict(model, tensor_state(authorized.checkpoint_path))
    config_authority.verify()
    return model


def build_manifest_authorized_model(
        repository_root: Path, manifest_path: Path,
        candidate: str | CandidateSpec, *,
        config_authority: ConfigAuthority,
        downstream_output: Path | None = None,
        materialized_authority_path: Path | None = None,
        materialized_config_path: Path | None = None,
        device: str = 'cpu') -> Any:
    """Derive the stage source, reconstruct it, and safely build the model."""
    root = lexical_repository_root(repository_root)
    candidate_id = candidate if isinstance(candidate, str) else candidate.id
    authorized = authorize_manifest_candidate(
        root, manifest_path, candidate_id)
    if isinstance(candidate, CandidateSpec) and authorized.candidate != candidate:
        raise ValueError('candidate differs from the authorized manifest entry')
    if not isinstance(config_authority, ConfigAuthority):
        raise TypeError('config_authority must be a ConfigAuthority locator')
    materialized_requested = (
        materialized_authority_path is not None
        or materialized_config_path is not None)
    if downstream_output is not None and materialized_requested:
        raise ValueError('model construction accepts exactly one stage source')
    if materialized_requested:
        if (
                materialized_authority_path is None
                or materialized_config_path is None):
            raise ValueError(
                'materialized model construction requires config and authority')
        config_relative = _repository_relative(
            root, materialized_config_path,
            label='materialized evaluation config')
        expected_directory = (
            Path('work_dirs/optimization') / authorized.candidate.route /
            authorized.candidate.id / str(authorized.candidate.seed) /
            'evaluate')
        if (
                config_relative.parent != expected_directory
                or config_relative.name not in {
                    'resolved-flip.py', 'resolved-no-flip.py'}):
            raise ValueError(
                'materialized evaluation config is not canonical for stage')
        expected_authority_relative = config_relative.with_name(
            f'{config_relative.stem}.config-authority.json')
        supplied_authority_relative = _repository_relative(
            root, materialized_authority_path,
            label='materialized config authority')
        if supplied_authority_relative != expected_authority_relative:
            raise ValueError(
                'materialized ConfigAuthority is not canonical for config')
        expected = load_materialized_config_authority(
            root, manifest_path, authorized.candidate,
            root / expected_authority_relative)
        if expected.path != root / config_relative:
            raise ValueError(
                'materialized ConfigAuthority config path differs from stage')
    elif downstream_output is not None:
        output_relative = _repository_relative(
            root, downstream_output, label='PWL downstream output')
        output_path = root / output_relative
        expected_stage_root = (
            root / 'work_dirs/optimization' / authorized.candidate.route /
            authorized.candidate.id / str(authorized.candidate.seed))
        try:
            stage_relative = output_path.relative_to(expected_stage_root)
        except ValueError as error:
            raise ValueError(
                'PWL downstream output is not canonical for candidate') from error
        if (
                len(stage_relative.parts) != 2
                or stage_relative.parts[0] not in {'profile', 'latency'}
                or stage_relative.name != f'{stage_relative.parts[0]}.json'):
            raise ValueError(
                'PWL downstream output does not identify a model stage')
        conversion_path = expected_stage_root / 'convert/convert.json'
        expected = authorize_pwl_runtime_config(
            root, manifest_path, authorized.candidate,
            conversion_path=conversion_path)
    else:
        expected = authorize_tracked_config(
            root, manifest_path, authorized.candidate)
    config_authority._require_exact(expected)
    return _build_authorized_model(
        authorized, config_authority=expected,
        repository_root=root, manifest_path=manifest_path,
        device=device)
