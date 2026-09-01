"""Fail-closed authority and CPU structural smoke for the combined candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

import torch
from torch import nn


COMBINED_CANDIDATE_ID = 'no-pif-pwl-softplus-s-v1'
COMBINED_AUTHORITY_PATH = Path(
    'optimization/no_pif_softplus_parent_authority.json')
COMBINED_PWL_ROLES = (
    'backbone.layers.0.blocks.0.op',
    'backbone.layers.1.blocks.0.op',
    'backbone.layers.2.blocks.0.op',
    'backbone.layers.2.blocks.1.op',
    'backbone.layers.3.blocks.0.op',
)
NO_PIF_SOURCE_COMMIT = 'd938530bad75fa6a8ac515b2c313792d8150da49'
PWL_SOURCE_COMMIT = 'f71fcff33bebe32241d0825eb2b736605a926183'


class CombinedCandidateAuthorityError(ValueError):
    """Raised when either independent parent or a claim limit changes."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob(root: Path, commit: str, path: str) -> bytes:
    if not isinstance(commit, str) or len(commit) != 40:
        raise CombinedCandidateAuthorityError('parent commit is invalid')
    if Path(path).is_absolute() or any(
            item in {'', '.', '..'} for item in Path(path).parts):
        raise CombinedCandidateAuthorityError('parent config path is unsafe')
    try:
        return subprocess.run(
            ['git', 'show', f'{commit}:{path}'], cwd=root, check=True,
            capture_output=True).stdout
    except subprocess.SubprocessError as error:
        raise CombinedCandidateAuthorityError(
            'parent source blob is unavailable') from error


def _canonical_authority() -> dict[str, Any]:
    return {
        'artifact_kind': 'no-pif-tail-aware-softplus-parent-authority',
        'candidate_id': COMBINED_CANDIDATE_ID,
        'claim_limits': {
            'combined_accuracy_measured': False,
            'fpga_speedup_claimed': False,
            'hardware_latency_claimed': False,
            'interaction_effect_measured': False,
            'isolated_deltas_additive': False,
        },
        'combination': {
            'pif_mode': 'disabled',
            'pwl_function': 'softplus',
            'pwl_role_count': 5,
            'pwl_roles': list(COMBINED_PWL_ROLES),
            'tail_policy': 'continuous-asymptotic-tail-v1',
        },
        'comparator': {
            'candidate_id': 'full-s-v1',
            'checkpoint': (
                'work_dirs/reproduction/runs/coco-s-v1/'
                'best_coco_AP_epoch_300.pth'),
            'checkpoint_sha256': (
                'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2'),
            'config': 'configs/reproduction/coco_s_v1.py',
            'config_sha256': (
                '7aa78c2b8394b57b16517380a479adb5c3b3161cadb802c26c2d2f66583a6b34'),
            'evaluation_artifact': {
                'path': (
                    'optimization/evidence_snapshots/no_pif_softplus/'
                    'full-s-v1-evaluate.json'),
                'sha256': (
                    'aab31a0bf41b7aa62c379e99c3818626705b95d01c07acf04c0adb26fefd4e5d'),
            },
            'evaluation_candidate_id': 'full-s-v1-frozen-r1',
            'source_commit': 'a6adf6d84f9b51b133b0fca7272c51f728bfb7ac',
        },
        'parents': {
            'no_pif': {
                'candidate_id': 'no-pif-s-v1',
                'checkpoint': (
                    'work_dirs/reproduction/runs/coco-s-v1-no-pif/'
                    'best_coco_AP_epoch_300.pth'),
                'checkpoint_sha256': (
                    '28cd02405e58d619896a0430f91936684084ef7e3423d507c7a10b743759a5fb'),
                'config': (
                    'configs/reproduction/ablations/coco_s_v1_no_pif.py'),
                'config_sha256': (
                    '706eaa316934d89a446510651d3d7dc3b34278436fa8caa47bdf3c228d1209e1'),
                'role': 'structural-and-checkpoint-parent',
                'source_commit': NO_PIF_SOURCE_COMMIT,
            },
            'tail_aware_softplus_pwl': {
                'candidate_id': 'pwl-softplus-s-v1',
                'checkpoint': (
                    'work_dirs/reproduction/runs/coco-s-v1/'
                    'best_coco_AP_epoch_300.pth'),
                'checkpoint_sha256': (
                    'a6f76dae86db4d92c445f26a428b61911e8b42c4de2119348997e9537cc7cdd2'),
                'config': 'configs/optimization/numeric/pwl_softplus.py',
                'config_sha256': (
                    '60ec03d3f74520e20694901ff257ee6a3ea8902460f7c34cffe2c189dc1595d3'),
                'isolated_evidence': {
                    'selection': {
                        'path': (
                            'optimization/evidence_snapshots/'
                            'no_pif_softplus/v6-selection.json'),
                        'sha256': (
                            'be8e07f4b65dcc95137c7357e7621192aff5252ef5a0ebd0888dbed3d7bb23e8'),
                    },
                    'installation': {
                        'path': (
                            'optimization/evidence_snapshots/'
                            'no_pif_softplus/v6-installation.json'),
                        'sha256': (
                            '0e8ecc484397922cc8b21326ca6581ffc042511a37d71db656d7a757ae7363ad'),
                    },
                    'smoke': {
                        'path': (
                            'optimization/evidence_snapshots/'
                            'no_pif_softplus/v6-smoke.json'),
                        'sha256': (
                            '6dcb4b8b52defd7fe4d087fd79b067d1651d1bf456fb9392b86be6df1ed6a666'),
                    },
                    'evaluate': {
                        'path': (
                            'optimization/evidence_snapshots/'
                            'no_pif_softplus/v6-evaluate.json'),
                        'sha256': (
                            '1726918fdd96402698b6b4d7b1ac347a45491d6df19a41631ff1ccdba908887e'),
                    },
                    'profile': {
                        'path': (
                            'optimization/evidence_snapshots/'
                            'no_pif_softplus/v6-profile.json'),
                        'sha256': (
                            '90c57b724e33bb685d39bc18a10f2a80be468081a48311eb9ec6515ee959c600'),
                    },
                    'latency': {
                        'path': (
                            'optimization/evidence_snapshots/'
                            'no_pif_softplus/v6-latency.json'),
                        'sha256': (
                            'e07516e26aa4c4eb2469836c8b6653f9cefcf042677084632c19affbf084fc94'),
                    },
                },
                'role': 'function-policy-and-isolated-evidence-parent',
                'source_commit': PWL_SOURCE_COMMIT,
            },
        },
        'schema_version': 1,
    }


def read_bound_workspace_artifact(
        reference: Mapping[str, Any], *, checkout_root: Path,
        ) -> dict[str, Any]:
    """Read one exact workspace artifact without following path symlinks."""
    if (not isinstance(reference, Mapping)
            or set(reference) != {'path', 'sha256'}
            or not isinstance(reference.get('path'), str)
            or not isinstance(reference.get('sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', reference['sha256'])):
        raise CombinedCandidateAuthorityError(
            'bound workspace artifact reference is invalid')
    relative = Path(reference['path'])
    if (relative.is_absolute() or not relative.parts
            or any(part in {'', '.', '..'} for part in relative.parts)):
        raise CombinedCandidateAuthorityError(
            'bound workspace artifact path is unsafe')
    root = Path(checkout_root).resolve(strict=True)
    cursor = root
    try:
        for part in relative.parts:
            cursor = cursor / part
            if stat.S_ISLNK(cursor.lstat().st_mode):
                raise CombinedCandidateAuthorityError(
                    'bound workspace artifact path uses symlink')
        directory_flags = (
            os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
            | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))
        directory = os.open(root, directory_flags)
        try:
            for part in relative.parts[:-1]:
                child = os.open(part, directory_flags, dir_fd=directory)
                os.close(directory)
                directory = child
            file_flags = (
                os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
                | getattr(os, 'O_NOFOLLOW', 0))
            descriptor = os.open(
                relative.parts[-1], file_flags, dir_fd=directory)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise CombinedCandidateAuthorityError(
                        'bound workspace artifact is not a regular file')
                blocks: list[bytes] = []
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    blocks.append(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(directory)
    except CombinedCandidateAuthorityError:
        raise
    except OSError as error:
        raise CombinedCandidateAuthorityError(
            'bound workspace artifact is unavailable') from error
    seal = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns)
    if seal(before) != seal(after):
        raise CombinedCandidateAuthorityError(
            'bound workspace artifact changed while being read')
    payload = b''.join(blocks)
    if _sha256_bytes(payload) != reference['sha256']:
        raise CombinedCandidateAuthorityError(
            'bound workspace artifact hash changed')
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CombinedCandidateAuthorityError(
            'bound workspace artifact is invalid JSON') from error
    if not isinstance(value, dict):
        raise CombinedCandidateAuthorityError(
            'bound workspace artifact root is not an object')
    return value


def _require_reference(
        actual: object, expected: Mapping[str, str], *, label: str) -> None:
    if actual != dict(expected):
        raise CombinedCandidateAuthorityError(
            f'frozen PWL {label} binding is invalid')


def _validate_frozen_pwl_artifacts(
        authority: Mapping[str, Any], *, repository_root: Path,
        ) -> dict[str, dict[str, Any]]:
    """Validate all six immutable v6 parent artifacts and cross-bind them."""
    evidence = authority['parents']['tail_aware_softplus_pwl'][
        'isolated_evidence']
    expected_names = (
        'selection', 'installation', 'smoke', 'evaluate', 'profile',
        'latency')
    if tuple(evidence) != expected_names:
        raise CombinedCandidateAuthorityError(
            'frozen PWL evidence inventory is incomplete or reordered')
    values = {
        name: read_bound_workspace_artifact(
            evidence[name], checkout_root=repository_root)
        for name in expected_names
    }
    parent = authority['parents']['tail_aware_softplus_pwl']
    candidate_id = parent['candidate_id']
    source_commit = parent['source_commit']
    selection = values['selection']
    if (selection.get('schema_version') != 2
            or selection.get('artifact_kind') !=
            'pwl-four-candidate-selection'
            or selection.get('decision') != 'selected'
            or selection.get('selected_candidate_id') != candidate_id
            or not isinstance(selection.get('candidates'), list)):
        raise CombinedCandidateAuthorityError(
            'frozen PWL selection artifact is invalid')
    rows = [row for row in selection['candidates']
            if isinstance(row, Mapping)
            and row.get('candidate_id') == candidate_id]
    if len(rows) != 1 or not isinstance(
            rows[0].get('calibration_artifact'), Mapping):
        raise CombinedCandidateAuthorityError(
            'frozen PWL selected fit binding is invalid')
    fit_reference = dict(rows[0]['calibration_artifact'])

    installation = values['installation']
    operation = installation.get('operation_manifest', {})
    if (installation.get('schema_version') != 2
            or installation.get('candidate_id') != candidate_id
            or installation.get('fit_artifact') != fit_reference
            or operation.get('function') != 'softplus'
            or operation.get('source') != 'ss2d-transition'
            or operation.get('operation_roles') != list(COMBINED_PWL_ROLES)
            or operation.get('segments_per_role') != 16
            or operation.get('hardware_latency_claimed') is not False):
        raise CombinedCandidateAuthorityError(
            'frozen PWL installation artifact is invalid')

    local_references = {
        'selection': {
            'path': ('work_dirs/optimization/ssm-quant-pwl/'
                     'pwl-selection/selection.json'),
            'sha256': evidence['selection']['sha256']},
        'installation': {
            'path': ('work_dirs/optimization/ssm-quant-pwl/'
                     'pwl-softplus-s-v1/0/convert/pwl-installation.json'),
            'sha256': evidence['installation']['sha256']},
        'smoke': {
            'path': ('work_dirs/optimization/ssm-quant-pwl/'
                     'pwl-softplus-s-v1/0/smoke-stage-a/smoke.json'),
            'sha256': evidence['smoke']['sha256']},
    }
    smoke = values['smoke']
    policy = smoke.get('policy', {})
    if (smoke.get('schema_version') != 2
            or smoke.get('artifact_kind') !=
            'pwl-stage-a-full-model-smoke'
            or smoke.get('candidate_id') != candidate_id
            or smoke.get('source') != {'git_commit': source_commit}
            or smoke.get('config') != {
                'path': parent['config'],
                'sha256': parent['config_sha256']}
            or smoke.get('checkpoint') != {
                'path': parent['checkpoint'],
                'sha256': parent['checkpoint_sha256']}
            or smoke.get('fit_artifact') != fit_reference
            or policy.get('enabled_function') != 'softplus'
            or policy.get('roles') != list(COMBINED_PWL_ROLES)):
        raise CombinedCandidateAuthorityError(
            'frozen PWL Stage-A artifact is invalid')
    _require_reference(
        smoke.get('selection_artifact'), local_references['selection'],
        label='selection')
    _require_reference(
        smoke.get('installation'), local_references['installation'],
        label='installation')

    from .evaluation import (
        MetricError, validate_evaluation_envelope,
        validate_latency_envelope)
    smoke_binding = local_references['smoke']
    evaluation = values['evaluate']
    evaluation_source = evaluation.get('result', {}).get('source')
    if not isinstance(evaluation_source, Mapping):
        raise CombinedCandidateAuthorityError(
            'frozen PWL evaluation source is invalid')
    try:
        validated_evaluation = validate_evaluation_envelope(
            evaluation, expected_candidate_id=candidate_id,
            expected_route='ssm-quant-pwl',
            expected_checkpoint_sha256=parent['checkpoint_sha256'],
            expected_source_config=(
                'work_dirs/optimization/ssm-quant-pwl/'
                'pwl-softplus-s-v1/0/convert/resolved-runtime.py'),
            expected_checkpoint=parent['checkpoint'], expected_seed=0,
            expected_git_commit=source_commit,
            expected_source_binding=evaluation_source,
            require_source_binding=True,
            expected_pwl_stage_a=smoke_binding)
    except MetricError as error:
        raise CombinedCandidateAuthorityError(
            f'frozen PWL evaluation artifact is invalid: {error}') from error

    profile = values['profile']
    expected_profile_fields = {
        'schema_version', 'git_commit', 'candidate', 'config', 'checkpoint',
        'checkpoint_sha256', 'input_shapes', 'output_shapes', 'parameters',
        'modules', 'device', 'parent', 'runtime', 'source', 'pwl_stage_a'}
    if (set(profile) != expected_profile_fields
            or profile.get('schema_version') != 2
            or profile.get('git_commit') != source_commit
            or profile.get('candidate') != candidate_id
            or profile.get('checkpoint') != parent['checkpoint']
            or profile.get('checkpoint_sha256') !=
            parent['checkpoint_sha256']
            or profile.get('pwl_stage_a') != smoke_binding
            or profile.get('parent') != {
                'config': parent['config'],
                'checkpoint': parent['checkpoint'],
                'checkpoint_sha256': parent['checkpoint_sha256']}
            or not isinstance(profile.get('modules'), list)
            or not profile['modules']
            or not isinstance(profile.get('parameters'), Mapping)):
        raise CombinedCandidateAuthorityError(
            'frozen PWL profile artifact is invalid')

    latency = values['latency']
    latency_source = latency.get('result', {}).get('source')
    if not isinstance(latency_source, Mapping):
        raise CombinedCandidateAuthorityError(
            'frozen PWL latency source is invalid')
    try:
        validated_latency = validate_latency_envelope(
            latency, expected_candidate_id=candidate_id,
            expected_route='ssm-quant-pwl',
            expected_checkpoint_sha256=parent['checkpoint_sha256'],
            expected_source_config=profile['config'],
            expected_checkpoint=parent['checkpoint'],
            expected_git_commit=source_commit,
            expected_config_sha256=profile['runtime']['config']['sha256'],
            expected_source_binding=latency_source,
            require_source_binding=True,
            expected_pwl_stage_a=smoke_binding)
    except MetricError as error:
        raise CombinedCandidateAuthorityError(
            f'frozen PWL latency artifact is invalid: {error}') from error
    return {
        'selection': {
            'selected_candidate_id': selection['selected_candidate_id']},
        'installation': {
            'candidate_id': installation['candidate_id'],
            'function': operation['function']},
        'smoke': {'candidate_id': smoke['candidate_id']},
        'evaluate': {
            'candidate_id': validated_evaluation['candidate_id']},
        'profile': {'candidate_id': profile['candidate']},
        'latency': {'candidate_id': candidate_id,
                    'route': validated_latency['route']},
    }


def validate_combined_parent_authority(
        value: Mapping[str, Any], *, candidate, repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    """Rebuild both source-parent bindings and the no-addition claim policy."""
    root = Path(repository_root).resolve(strict=True)
    if not isinstance(value, Mapping):
        raise CombinedCandidateAuthorityError(
            'combined parent authority differs from the canonical contract')
    supplied = dict(value)
    supplied.pop('validated_parent_artifacts', None)
    supplied.pop('validated_comparator_artifact', None)
    if supplied != _canonical_authority():
        raise CombinedCandidateAuthorityError(
            'combined parent authority differs from the canonical contract')
    if candidate.id != COMBINED_CANDIDATE_ID or candidate.features != {
            'numeric_kind': 'pwl-combined',
            'pwl_function': 'softplus',
            'pif_mode': 'disabled',
            'calibration_source_candidate': 'no-pif-s-v1',
            'combined_parent_authority': COMBINED_AUTHORITY_PATH.as_posix(),
            'auto_run': False,
            'conditional': True,
            'prune_disabled_pif': True}:
        raise CombinedCandidateAuthorityError(
            'combined candidate manifest contract is invalid')
    from .schema import load_candidate_manifest
    by_id = {item.id: item for item in load_candidate_manifest(manifest_path)}
    if set(('full-s-v1', 'no-pif-s-v1', 'pwl-softplus-s-v1')) - set(by_id):
        raise CombinedCandidateAuthorityError('combined parents are missing')
    no_pif = value['parents']['no_pif']
    full = value['comparator']
    source = by_id['no-pif-s-v1']
    comparator = by_id['full-s-v1']
    pwl = by_id['pwl-softplus-s-v1']
    pwl_parent = value['parents']['tail_aware_softplus_pwl']
    if (source.config.as_posix() != no_pif['config']
            or source.checkpoint.as_posix() != no_pif['checkpoint']
            or source.checkpoint_sha256 != no_pif['checkpoint_sha256']
            or candidate.checkpoint != source.checkpoint
            or candidate.checkpoint_sha256 != source.checkpoint_sha256
            or comparator.config.as_posix() != full['config']
            or comparator.checkpoint.as_posix() != full['checkpoint']
            or comparator.checkpoint_sha256 != full['checkpoint_sha256']
            or pwl.config.as_posix() != pwl_parent['config']
            or pwl.checkpoint.as_posix() != pwl_parent['checkpoint']
            or pwl.checkpoint_sha256 != pwl_parent['checkpoint_sha256']):
        raise CombinedCandidateAuthorityError(
            'combined checkpoint/comparator lineage is invalid')
    for parent in (no_pif, supplied['parents']['tail_aware_softplus_pwl'],
                   full):
        payload = _git_blob(root, parent['source_commit'], parent['config'])
        if _sha256_bytes(payload) != parent['config_sha256']:
            raise CombinedCandidateAuthorityError(
                'parent config hash disagrees with source commit')
        live = root / parent['config']
        if not live.is_file() or live.is_symlink() or live.read_bytes() != payload:
            raise CombinedCandidateAuthorityError(
                'live parent config differs from source commit')
    validated_parent_artifacts = _validate_frozen_pwl_artifacts(
        supplied, repository_root=root)
    comparator_evaluation = read_bound_workspace_artifact(
        full['evaluation_artifact'], checkout_root=root)
    from .combined_comparison import validate_comparator_evaluation
    validate_comparator_evaluation(
        comparator_evaluation, comparator=full)
    result = copy.deepcopy(supplied)
    result['validated_parent_artifacts'] = validated_parent_artifacts
    result['validated_comparator_artifact'] = {
        'candidate_id': comparator_evaluation['candidate_id'],
        'path': full['evaluation_artifact']['path'],
        'sha256': full['evaluation_artifact']['sha256'],
    }
    return result


def load_combined_parent_authority(
        candidate, *, repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    relative = Path(candidate.features.get('combined_parent_authority', ''))
    if relative != COMBINED_AUTHORITY_PATH:
        raise CombinedCandidateAuthorityError(
            'combined authority path is not canonical')
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise CombinedCandidateAuthorityError(
            'combined authority is missing or uses symlink')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise CombinedCandidateAuthorityError(
            'combined authority is invalid JSON') from error
    return validate_combined_parent_authority(
        value, candidate=candidate, repository_root=root,
        manifest_path=manifest_path)


def is_combined_pwl(candidate) -> bool:
    return getattr(candidate, 'features', {}).get(
        'numeric_kind') == 'pwl-combined'


def load_pwl_admission_reference(
        reference: Mapping[str, Any], *, repository_root: Path,
        manifest_path: Path) -> dict[str, Any]:
    """Validate either canonical four-way selection or combined parents."""
    if (not isinstance(reference, Mapping)
            or set(reference) != {'path', 'sha256'}
            or not isinstance(reference.get('path'), str)
            or not isinstance(reference.get('sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', reference['sha256'])):
        raise CombinedCandidateAuthorityError(
            'PWL admission reference is invalid')
    root = Path(repository_root).resolve(strict=True)
    relative = Path(reference['path'])
    if (relative.is_absolute()
            or any(item in {'', '.', '..'} for item in relative.parts)):
        raise CombinedCandidateAuthorityError('PWL admission path is unsafe')
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise CombinedCandidateAuthorityError('PWL admission is missing')
    actual = _sha256_bytes(path.read_bytes())
    if actual != reference['sha256']:
        raise CombinedCandidateAuthorityError('PWL admission hash changed')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise CombinedCandidateAuthorityError(
            'PWL admission is invalid JSON') from error
    if value.get('artifact_kind') == (
            'no-pif-tail-aware-softplus-parent-authority'):
        if relative != COMBINED_AUTHORITY_PATH:
            raise CombinedCandidateAuthorityError(
                'combined admission path is not canonical')
        from .schema import load_candidate_manifest
        matches = tuple(
            item for item in load_candidate_manifest(manifest_path)
            if item.id == value.get('candidate_id'))
        if len(matches) != 1:
            raise CombinedCandidateAuthorityError(
                'combined admission candidate is missing')
        authority = validate_combined_parent_authority(
            value, candidate=matches[0], repository_root=root,
            manifest_path=manifest_path)
        return {
            'admission_kind': 'combined-parent-authority-v1',
            'decision': 'selected',
            'selected_candidate_id': authority['candidate_id'],
            'authority': authority,
        }
    from .pwl_selection import load_pwl_selection_reference
    return load_pwl_selection_reference(
        reference, repository_root=root, manifest_path=manifest_path)


class _PrunedDisabledPoseInteraction(nn.Module):
    """Parameter-free identity left after authorized no-PIF state loading."""

    mode = 'disabled'

    def __init__(self, *, dim: int, num_keypoints: int):
        super().__init__()
        self.dim = int(dim)
        self.num_keypoints = int(num_keypoints)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (
                self.num_keypoints, self.dim):
            raise ValueError(
                'disabled PIF expects [batch, num_keypoints, dim], got '
                f'{tuple(tokens.shape)}')
        return tokens


class _SmokeSoftplusPWL(nn.Module):
    """CPU-only topology probe with the admitted continuous Softplus tails."""

    function_name = 'softplus'
    saturation = 'continuous-asymptotic-tail-v1'

    def __init__(self):
        super().__init__()
        points = torch.linspace(-8.0, 8.0, 17, dtype=torch.float64)
        values = torch.nn.functional.softplus(points)
        slopes = (values[1:] - values[:-1]) / (points[1:] - points[:-1])
        intercepts = values[:-1] - slopes * points[:-1]
        self.register_buffer('breakpoints', points)
        self.register_buffer('slopes', slopes)
        self.register_buffer('intercepts', intercepts)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        points = self.breakpoints.to(device=value.device, dtype=value.dtype)
        slopes = self.slopes.to(device=value.device, dtype=value.dtype)
        intercepts = self.intercepts.to(device=value.device, dtype=value.dtype)
        bounded = value.clamp(points[0], points[-1])
        index = torch.bucketize(bounded, points[1:-1], right=True)
        interior = slopes[index] * bounded + intercepts[index]
        lower = slopes[0] * points[0] + intercepts[0]
        upper = slopes[-1] * points[-1] + intercepts[-1]
        return torch.where(
            value < points[0], lower,
            torch.where(value > points[-1], value + upper - points[-1],
                        interior))


def _parent_and_leaf(model: nn.Module, role: str) -> tuple[nn.Module, str]:
    path = role.split('.')
    parent = model
    for part in path[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, path[-1]


def prune_disabled_pif(model: nn.Module) -> tuple[str, ...]:
    """Remove bypassed PIF state only after the checkpoint was strictly loaded."""
    prior = getattr(model, '_combined_pruned_pif_paths', None)
    if prior is not None:
        return tuple(prior)
    matches = tuple(
        (name, module) for name, module in model.named_modules()
        if name.endswith('pose_interaction')
        and getattr(module, 'mode', None) == 'disabled')
    if len(matches) != 1:
        raise CombinedCandidateAuthorityError(
            'combined model must expose exactly one disabled PIF module')
    name, module = matches[0]
    parent, leaf = _parent_and_leaf(model, name)
    setattr(parent, leaf, _PrunedDisabledPoseInteraction(
        dim=getattr(module, 'dim', 256),
        num_keypoints=getattr(module, 'num_keypoints', 17)))
    model._combined_pruned_pif_paths = (name,)
    return (name,)


def install_combined_cpu_smoke(
        model: nn.Module, *, roles: Sequence[str], function_name: str,
        domain: tuple[float, float], segments: int,
        saturation: str) -> dict[str, Any]:
    """Install an unmeasured fit only to prove the combined runtime topology."""
    if (tuple(roles) != COMBINED_PWL_ROLES or function_name != 'softplus'
            or tuple(domain) != (-8.0, 8.0) or segments != 16
            or saturation != 'continuous-asymptotic-tail-v1'):
        raise CombinedCandidateAuthorityError(
            'combined CPU smoke policy differs from the admitted topology')
    removed = prune_disabled_pif(model)
    approximation = _SmokeSoftplusPWL()
    modules = dict(model.named_modules())
    if any(role not in modules for role in roles):
        raise CombinedCandidateAuthorityError('combined PWL role is missing')
    for role in roles:
        target = modules[role]
        installer = getattr(target, 'install_numeric_pwl', None)
        if not callable(installer):
            raise CombinedCandidateAuthorityError(
                'combined PWL role is not an SS2D installation target')
        installer(function_name, copy.deepcopy(approximation))
    installed = tuple(
        name for name, module in model.named_modules()
        if getattr(module, '_numeric_pwl_function', None) == function_name)
    if installed != COMBINED_PWL_ROLES:
        raise CombinedCandidateAuthorityError(
            'combined CPU smoke installed an unexpected PWL topology')
    return {
        'pif_modules_removed': list(removed),
        'pif_mode': 'disabled',
        'pwl_function': function_name,
        'pwl_roles': list(installed),
        'pwl_role_count': len(installed),
        'tail_policy': saturation,
    }


__all__ = [
    'COMBINED_CANDIDATE_ID', 'COMBINED_PWL_ROLES',
    'CombinedCandidateAuthorityError', 'install_combined_cpu_smoke',
    'is_combined_pwl', 'load_combined_parent_authority',
    'load_pwl_admission_reference', 'prune_disabled_pif',
    'validate_combined_parent_authority']
