"""Fail-closed authority for the interpreter used by formal Stage C.

This module deliberately imports only the Python standard library.  Torch and
other native packages are inventory data, never imports in the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Mapping


class FormalEnvironmentError(ValueError):
    """The current process environment does not match its authority."""


_REQUIRED_ENVIRONMENT = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'PYTHONNOUSERSITE': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'CUDA_VISIBLE_DEVICES': '0',
    'MAMBAPOSE_PHYSICAL_DEVICE_INDEX': '0',
    'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
}
_STRIPPED_ENVIRONMENT = (
    'PYTHONPATH',
    'PYTHONHOME',
    'PYTHONSTARTUP',
    'PYTHONUSERBASE',
    'LD_PRELOAD',
    'LD_LIBRARY_PATH',
    'TORCH_FORCE_WEIGHTS_ONLY_LOAD',
)
_REQUIREMENTS = (
    ('requirements', 'requirements.txt'),
    ('runtime', 'requirements/runtime.txt'),
    ('reproduction_constraints', 'requirements/reproduction-constraints.txt'),
)
_NATIVE_SUFFIXES = ('.so', '.pyd', '.dll', '.dylib')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        + '\n').encode('ascii')


def _sanitized_process_environment(
        environment: Mapping[str, str]) -> dict[str, str]:
    """Remove ambient import/linker authority and fix executable lookup."""
    result = dict(environment)
    unsafe_override = result.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
    if unsafe_override == '1':
        raise FormalEnvironmentError(
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is forbidden')
    for key in _STRIPPED_ENVIRONMENT:
        result.pop(key, None)
    result['PATH'] = _REQUIRED_ENVIRONMENT['PATH']
    return result


def _lexical_relative(value: str, *, label: str) -> Path:
    if not isinstance(value, str) or not value or '\\' in value:
        raise FormalEnvironmentError(f'{label} must be a canonical path')
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {'', '.', '..'} for part in pure.parts):
        raise FormalEnvironmentError(f'{label} must be a canonical path')
    if pure.as_posix() != value:
        raise FormalEnvironmentError(f'{label} must be a canonical path')
    return Path(value)


@dataclass(frozen=True, order=True)
class RequirementBinding:
    role: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {'path': self.path, 'role': self.role, 'sha256': self.sha256}


@dataclass(frozen=True, order=True)
class NativeModuleBinding:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, str | int]:
        return {'path': self.path, 'sha256': self.sha256, 'size': self.size}


@dataclass(frozen=True)
class EnvironmentAuthority:
    schema_version: int
    venv_link: str
    venv_target: str
    interpreter_path: str
    interpreter_real_path: str
    interpreter_sha256: str
    requirements: tuple[RequirementBinding, ...]
    packages: tuple[str, ...]
    native_modules: tuple[NativeModuleBinding, ...]
    required_environment: tuple[tuple[str, str], ...]
    inventory_sha256: str

    @classmethod
    def capture(cls, repository_root: Path) -> 'EnvironmentAuthority':
        root = Path(repository_root)
        if not root.is_absolute():
            root = root.resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise FormalEnvironmentError(
                'repository root must be an existing non-symlink directory')
        venv_link = root / '.venv'
        if not venv_link.is_symlink():
            raise FormalEnvironmentError('.venv must be an explicit symlink')
        target = venv_link.resolve(strict=True)
        if not target.is_dir():
            raise FormalEnvironmentError('.venv target must be a directory')
        interpreter = venv_link / 'bin/python'
        real_interpreter = interpreter.resolve(strict=True)
        if not real_interpreter.is_file():
            raise FormalEnvironmentError('environment interpreter is missing')

        probe_environment = _sanitized_process_environment(os.environ)
        probe_environment.update({
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONNOUSERSITE': '1',
        })
        probe = subprocess.run(
            [str(interpreter), '-B', '-c', _PROBE],
            cwd=root,
            env=probe_environment,
            check=True, text=True, capture_output=True)
        try:
            observed = json.loads(probe.stdout)
        except json.JSONDecodeError as error:
            raise FormalEnvironmentError(
                'environment inventory probe was not valid JSON') from error
        if set(observed) != {'packages', 'site_packages'}:
            raise FormalEnvironmentError(
                'environment inventory probe fields are invalid')
        packages = tuple(observed['packages'])
        if not packages or packages != tuple(sorted(set(packages))):
            raise FormalEnvironmentError('package inventory is not canonical')

        native: list[NativeModuleBinding] = []
        for raw_site in observed['site_packages']:
            site = Path(raw_site).resolve(strict=True)
            try:
                site.relative_to(target)
            except ValueError as error:
                raise FormalEnvironmentError(
                    'site-packages escapes the environment target') from error
            if site.is_symlink() or not site.is_dir():
                raise FormalEnvironmentError('site-packages is not authoritative')
            for path in sorted(
                    candidate for candidate in site.rglob('*')
                    if candidate.is_file()
                    and any(candidate.name.endswith(suffix)
                            for suffix in _NATIVE_SUFFIXES)):
                if path.is_symlink():
                    raise FormalEnvironmentError(
                        'native module inventory contains a symlink')
                relative = path.relative_to(target).as_posix()
                native.append(NativeModuleBinding(
                    relative, _sha256(path), path.stat().st_size))
        native_modules = tuple(sorted(set(native)))

        requirements = tuple(RequirementBinding(
            role=role, path=relative, sha256=_sha256(root / relative))
            for role, relative in _REQUIREMENTS)
        base = {
            'schema_version': 1,
            'venv_link': '.venv',
            'venv_target': str(target),
            'interpreter_path': '.venv/bin/python',
            'interpreter_real_path': str(real_interpreter),
            'interpreter_sha256': _sha256(real_interpreter),
            'requirements': [item.to_dict() for item in requirements],
            'packages': list(packages),
            'native_modules': [item.to_dict() for item in native_modules],
            'required_environment': dict(sorted(_REQUIRED_ENVIRONMENT.items())),
        }
        inventory_sha256 = hashlib.sha256(_canonical_bytes(base)).hexdigest()
        return cls(
            schema_version=1,
            venv_link='.venv',
            venv_target=str(target),
            interpreter_path='.venv/bin/python',
            interpreter_real_path=str(real_interpreter),
            interpreter_sha256=base['interpreter_sha256'],
            requirements=requirements,
            packages=packages,
            native_modules=native_modules,
            required_environment=tuple(sorted(_REQUIRED_ENVIRONMENT.items())),
            inventory_sha256=inventory_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'venv_link': self.venv_link,
            'venv_target': self.venv_target,
            'interpreter_path': self.interpreter_path,
            'interpreter_real_path': self.interpreter_real_path,
            'interpreter_sha256': self.interpreter_sha256,
            'requirements': [item.to_dict() for item in self.requirements],
            'packages': list(self.packages),
            'native_modules': [item.to_dict() for item in self.native_modules],
            'required_environment': dict(self.required_environment),
            'inventory_sha256': self.inventory_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'EnvironmentAuthority':
        if not isinstance(value, Mapping):
            raise FormalEnvironmentError('environment authority must be an object')
        fields = {
            'schema_version', 'venv_link', 'venv_target', 'interpreter_path',
            'interpreter_real_path', 'interpreter_sha256', 'requirements',
            'packages', 'native_modules', 'required_environment',
            'inventory_sha256',
        }
        if set(value) != fields:
            raise FormalEnvironmentError('environment authority fields mismatch')
        try:
            requirements = tuple(RequirementBinding(**item)
                                 for item in value['requirements'])
            native = tuple(NativeModuleBinding(**item)
                           for item in value['native_modules'])
            environment = tuple(sorted(value['required_environment'].items()))
            authority = cls(
                schema_version=value['schema_version'],
                venv_link=value['venv_link'],
                venv_target=value['venv_target'],
                interpreter_path=value['interpreter_path'],
                interpreter_real_path=value['interpreter_real_path'],
                interpreter_sha256=value['interpreter_sha256'],
                requirements=requirements,
                packages=tuple(value['packages']),
                native_modules=native,
                required_environment=environment,
                inventory_sha256=value['inventory_sha256'],
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise FormalEnvironmentError(
                'environment authority field types are invalid') from error
        authority._validate_structure()
        return authority

    def _validate_structure(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise FormalEnvironmentError('environment schema_version must be 1')
        if self.venv_link != '.venv':
            raise FormalEnvironmentError('venv link must be .venv')
        _lexical_relative(self.venv_link, label='venv link')
        if self.interpreter_path != '.venv/bin/python':
            raise FormalEnvironmentError('interpreter path is not canonical')
        _lexical_relative(self.interpreter_path, label='interpreter path')
        for value, label in (
                (self.interpreter_sha256, 'interpreter'),
                (self.inventory_sha256, 'inventory')):
            if not isinstance(value, str) or len(value) != 64 \
                    or any(character not in '0123456789abcdef'
                           for character in value):
                raise FormalEnvironmentError(f'{label} SHA-256 is invalid')
        if not Path(self.venv_target).is_absolute() \
                or not Path(self.interpreter_real_path).is_absolute():
            raise FormalEnvironmentError('resolved environment paths must be absolute')
        if self.requirements != tuple(sorted(self.requirements)):
            # Canonical role order is intentionally not alphabetical.
            if tuple(item.role for item in self.requirements) != tuple(
                    item[0] for item in _REQUIREMENTS):
                raise FormalEnvironmentError('requirements are not canonical')
        if tuple(item.role for item in self.requirements) != tuple(
                item[0] for item in _REQUIREMENTS):
            raise FormalEnvironmentError('requirements roles mismatch')
        if tuple(item.path for item in self.requirements) != tuple(
                item[1] for item in _REQUIREMENTS):
            raise FormalEnvironmentError('requirements paths mismatch')
        if self.packages != tuple(sorted(set(self.packages))):
            raise FormalEnvironmentError('packages are not canonical')
        if self.native_modules != tuple(sorted(set(self.native_modules))):
            raise FormalEnvironmentError('native modules are not canonical')
        if dict(self.required_environment) != _REQUIRED_ENVIRONMENT:
            raise FormalEnvironmentError('required environment values mismatch')
        document = self.to_dict()
        supplied = document.pop('inventory_sha256')
        observed = hashlib.sha256(_canonical_bytes(document)).hexdigest()
        if supplied != observed:
            raise FormalEnvironmentError('environment inventory SHA-256 mismatch')


def validate_environment_authority(
        authority: EnvironmentAuthority, repository_root: Path) -> None:
    if not isinstance(authority, EnvironmentAuthority):
        raise FormalEnvironmentError('environment authority type is invalid')
    authority._validate_structure()
    observed = EnvironmentAuthority.capture(Path(repository_root))
    if observed != authority:
        raise FormalEnvironmentError('environment authority differs from observation')


def apply_required_process_environment(
        authority: EnvironmentAuthority,
        environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an exec environment, setting absent values and rejecting drift."""
    authority._validate_structure()
    result = _sanitized_process_environment(
        os.environ if environment is None else environment)
    for key, expected in authority.required_environment:
        observed = result.get(key)
        if observed is not None and observed != expected:
            raise FormalEnvironmentError(
                f'process environment {key} conflicts with authority')
        result[key] = expected
    return result


_PROBE = r'''
import importlib.metadata
import json
import site

packages = sorted({
    ((distribution.metadata.get('Name') or distribution.name).lower()
     + '==' + distribution.version)
    for distribution in importlib.metadata.distributions()
})
site_packages = sorted(set(site.getsitepackages()))
print(json.dumps({'packages': packages, 'site_packages': site_packages},
                 sort_keys=True, separators=(',', ':')))
'''
