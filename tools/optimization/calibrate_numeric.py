#!/usr/bin/env python3
"""Calibrate Route 3 only from manifest-bound full S-V1 train2017 data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.checkpoints import authorize_manifest_candidate
from mambapose_opt.determinism import seed_deterministic_root
from mambapose_opt.numeric_calibration import (
    CalibrationTargets, calibration_identity, discover_calibration_targets,
    validate_calibration_artifact)
from mambapose_opt.schema import load_candidate_manifest
from mambapose_opt.numeric_source import build_numeric_source_binding
from mambapose_opt.pwl_artifacts import PWLObservationAccumulator
from mambapose_opt.source import clean_git_commit
from mmpose.models.utils.hardware_friendly import ActivationRangeObserver


def _candidate(path: Path, identifier: str):
    candidates = tuple(
        candidate for candidate in load_candidate_manifest(path)
        if candidate.id == identifier)
    if len(candidates) != 1:
        raise ValueError(f'candidate must resolve exactly once: {identifier}')
    return candidates[0]


def _git_commit(*, require_clean: bool) -> str:
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY_ROOT,
        text=True).strip()
    if require_clean:
        return clean_git_commit(REPOSITORY_ROOT)
    return commit


def _identity(candidate, policy: Path) -> dict[str, Any]:
    value = calibration_identity(
        repository_root=REPOSITORY_ROOT,
        candidate_id=candidate.id,
        config=candidate.config,
        checkpoint=candidate.checkpoint,
        expected_checkpoint_sha256=candidate.checkpoint_sha256,
        policy=policy.relative_to(REPOSITORY_ROOT),
        split='train2017',
        annotation=Path(
            'data/coco/annotations/person_keypoints_train2017.json'),
        image_prefix=Path('data/coco/train2017'))
    value['git_commit'] = _git_commit(require_clean=False)
    return value


def _require_target_policy(target_candidate, policy: Path) -> Path:
    """Bind calibration policy input to the target candidate config."""
    root = REPOSITORY_ROOT.resolve()
    try:
        expected = (root / target_candidate.config).resolve(strict=True)
        supplied = (
            policy if policy.is_absolute() else root / policy
        ).resolve(strict=True)
        expected.relative_to(root)
        supplied.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(
            'calibration policy must equal target candidate config') from error
    if supplied != expected:
        raise ValueError(
            'calibration policy must equal target candidate config')
    return expected


def _target_dict(targets: CalibrationTargets) -> dict[str, list[str]]:
    return {
        name: list(getattr(targets, name))
        for name in targets.__dataclass_fields__
    }


def audit(
        candidate, policy: Path, *, manifest_path: Path | None = None,
        target_candidate=None,
        ) -> dict[str, Any]:
    """CPU checkpoint/config load audit; no dataset iteration or artifact write."""
    target = target_candidate or candidate
    policy = _require_target_policy(target, policy)
    from mmpose.apis import init_model

    manifest = manifest_path or REPOSITORY_ROOT / 'optimization/candidates.json'
    authorized = authorize_manifest_candidate(
        REPOSITORY_ROOT, manifest, candidate.id)
    if authorized.candidate != candidate:
        raise ValueError('calibration candidate differs from authorized manifest')
    identity = _identity(candidate, policy)
    model = init_model(
        str(authorized.config_path),
        str(authorized.checkpoint_path),
        device='cpu')
    model.eval()
    targets = discover_calibration_targets(model)
    return {
        'status': 'audit-only',
        'identity': identity,
        'model_mode': 'eval',
        'grad_enabled': False,
        'targets': _target_dict(targets),
    }


def _tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for child in value if (item := _tensor(child)) is not None),
                    None)
    if isinstance(value, dict):
        return next((item for child in value.values()
                     if (item := _tensor(child)) is not None), None)
    return None


class _HookSession:
    def __init__(
            self, model: torch.nn.Module, targets: CalibrationTargets,
            *, pwl_policy: Mapping[str, Any] | None = None):
        self.model = model
        self.targets = targets
        self.observers: dict[str, ActivationRangeObserver] = {}
        self.handles = []
        self.functional_modules = []
        self.pwl = (
            PWLObservationAccumulator(pwl_policy)
            if pwl_policy is not None else None)

    def _record(self, name: str, value: Any, granularity: str = 'tensor') -> None:
        tensor = _tensor(value)
        if tensor is None:
            raise ValueError(f'calibration hook {name} did not produce a tensor')
        observer = self.observers.setdefault(
            name, ActivationRangeObserver(granularity))
        observer(tensor.detach())

    def __enter__(self):
        modules = dict(self.model.named_modules())
        if self.pwl is not None and self.pwl.policy['source'] == 'module':
            missing = tuple(
                role for role in self.pwl.policy['roles']
                if role not in modules)
            if missing:
                raise ValueError(f'PWL calibration roles are missing: {missing}')
            expected = (
                torch.nn.SiLU
                if self.pwl.policy['enabled_function'] == 'silu'
                else torch.nn.GELU)
            invalid = tuple(
                role for role in self.pwl.policy['roles']
                if type(modules[role]) is not expected)
            if invalid:
                raise ValueError(
                    f'PWL calibration module roles are incompatible: {invalid}')
            for role in self.pwl.policy['roles']:
                self.handles.append(modules[role].register_forward_pre_hook(
                    lambda _module, inputs, operation_role=role:
                    self.pwl.observe(operation_role, _tensor(inputs))))
        for name in self.targets.ss2d_boundaries:
            module = modules[name]
            self.handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, role=f'{name}.input':
                self._record(role, inputs)))
            self.handles.append(module.register_forward_hook(
                lambda _module, _inputs, output, role=f'{name}.output':
                self._record(role, output)))
        for name in self.targets.vmamba_in_proj + self.targets.vmamba_out_proj:
            self.handles.append(modules[name].register_forward_pre_hook(
                lambda _module, inputs, role=f'{name}.input':
                self._record(role, inputs)))
            self.handles.append(modules[name].register_forward_hook(
                lambda _module, _inputs, output, role=name:
                self._record(role, output, 'channel')))
        for name in self.targets.attention_qkv:
            self.handles.append(modules[name].register_forward_pre_hook(
                lambda _module, inputs, role=f'{name}.input':
                self._record(role, inputs)))
            def qkv_hook(_module, _inputs, output, role=name):
                if not isinstance(output, torch.Tensor) or output.shape[-1] % 3:
                    raise ValueError(f'{role} Q/K/V output is not divisible by three')
                for suffix, tensor in zip(('q', 'k', 'v'), output.chunk(3, -1)):
                    self._record(f'{role}.{suffix}', tensor, 'token')
            self.handles.append(modules[name].register_forward_hook(qkv_hook))
        for name in self.targets.pif_boundaries:
            module = modules[name]
            self.handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, role=f'{name}.input':
                self._record(role, inputs, 'token')))
            self.handles.append(module.register_forward_hook(
                lambda _module, _inputs, output, role=f'{name}.output':
                self._record(role, output, 'token')))
        for name in self.targets.heatmap_projection:
            self.handles.append(modules[name].register_forward_pre_hook(
                lambda _module, inputs, role=f'{name}.input':
                self._record(role, inputs)))
            self.handles.append(modules[name].register_forward_hook(
                lambda _module, _inputs, output, role=name:
                self._record(role, output, 'channel')))
        for name in self.targets.functional_observers:
            module = modules[name]
            module.set_numeric_observer(
                lambda role, value, prefix=name:
                self._functional_record(prefix, role, value))
            self.functional_modules.append(module)
        parameters = dict(self.model.named_parameters())
        for name in self.targets.transition_parameters:
            self._record(name, parameters[name])
        return self

    def __exit__(self, *_args):
        for module in self.functional_modules:
            module.set_numeric_observer(None)
        for handle in self.handles:
            handle.remove()

    def records(self) -> dict[str, dict[str, Any]]:
        return {name: observer.summary() for name, observer in self.observers.items()}

    def _functional_record(
            self, prefix: str, role: str, value: torch.Tensor) -> None:
        exact = {'transition_exp_input', 'transition_softplus_input'}
        if role in exact:
            if (self.pwl is not None
                    and self.pwl.policy['source'] == 'ss2d-transition'
                    and self.pwl.policy['enabled_function'] in role
                    and prefix in self.pwl.policy['roles']):
                self.pwl.observe(prefix, value)
            return
        self._record(f'{prefix}.{role}', value)

    def pwl_report(self, *, candidate_id: str) -> dict[str, Any]:
        if self.pwl is None:
            raise ValueError('PWL calibration policy is not installed')
        return self.pwl.report(candidate_id=candidate_id)


def _required_records(targets: CalibrationTargets) -> tuple[str, ...]:
    functional_roles = (
        'x_proj', 'dt_proj', 'scan_input_u', 'scan_input_dt',
        'transition_A', 'transition_B', 'transition_C', 'transition_D',
        'transition_delta_bias', 'scan_output')
    return (
        tuple(f'{name}.{side}' for name in targets.ss2d_boundaries
              for side in ('input', 'output'))
        + targets.vmamba_in_proj + targets.vmamba_out_proj
        + tuple(f'{name}.input' for name in (
            targets.vmamba_in_proj + targets.vmamba_out_proj))
        + tuple(f'{name}.{role}' for name in targets.attention_qkv
                for role in ('q', 'k', 'v'))
        + tuple(f'{name}.input' for name in targets.attention_qkv)
        + tuple(f'{name}.{side}' for name in targets.pif_boundaries
                for side in ('input', 'output'))
        + targets.heatmap_projection
        + tuple(f'{name}.input' for name in targets.heatmap_projection)
        + targets.transition_parameters
        + tuple(f'{name}.{role}' for name in targets.functional_observers
                for role in functional_roles)
    )


def _sample_ids(batch: Any) -> tuple[str, ...]:
    if not isinstance(batch, dict):
        raise ValueError('calibration dataloader batch must be a mapping')
    samples = batch.get('data_samples')
    if not isinstance(samples, (tuple, list)) or not samples:
        raise ValueError('calibration batch has no data_samples identity')
    result = []
    for sample in samples:
        identifier = getattr(sample, 'img_id', None)
        if identifier is None and hasattr(sample, 'metainfo'):
            identifier = sample.metainfo.get('img_id')
        if identifier is None:
            raise ValueError('calibration sample has no img_id')
        result.append(str(identifier))
    return tuple(result)


def _w8a8_activation_scales_from_records(
        activation_observers: Mapping[str, str],
        records: Mapping[str, Mapping[str, Any]],
        ) -> dict[str, dict[str, Any]]:
    """Derive W8A8 tensor scales from exact extrema, not histogram bounds."""
    activation_scales = {}
    for role, source_record in activation_observers.items():
        summary = records.get(source_record)
        if not isinstance(summary, Mapping):
            raise ValueError(
                f'activation observer record is missing for {role}: '
                f'{source_record}')
        numeric_range = summary.get('range')
        if (not isinstance(numeric_range, list) or len(numeric_range) != 2
                or any(not isinstance(item, (int, float))
                       or isinstance(item, bool) or not math.isfinite(item)
                       for item in numeric_range)):
            raise ValueError(
                f'activation observer has no finite measured range: {role}')
        overflow = summary.get('overflow_count')
        if (not isinstance(overflow, int) or isinstance(overflow, bool)
                or overflow < 0):
            raise ValueError(
                f'activation observer overflow count is invalid: {role}')
        if overflow:
            raise ValueError(
                f'activation observer has histogram overflow: {role}')
        if summary.get('granularity') != 'tensor':
            raise ValueError(
                f'activation observer granularity is not tensor: {role}')
        maximum = max(abs(float(item)) for item in numeric_range)
        scale = maximum / 127.0
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f'activation observer has no positive max-abs scale: {role}')
        activation_scales[role] = {
            'source_record': source_record,
            'granularity': 'tensor',
            'scale': scale,
        }
    return activation_scales


def calibrate(
        candidate, policy: Path, *, samples: int, device: str,
        target_candidate=None, manifest_path: Path | None = None) -> dict:
    if device != 'cuda:0':
        raise ValueError('production numeric calibration requires cuda:0')
    if samples <= 0 or samples > 4096:
        raise ValueError('calibration samples must be in [1, 4096]')
    target = target_candidate or candidate
    policy = _require_target_policy(target, policy)
    manifest = manifest_path or REPOSITORY_ROOT / 'optimization/candidates.json'
    authorized = authorize_manifest_candidate(
        REPOSITORY_ROOT, manifest, candidate.id)
    if authorized.candidate != candidate:
        raise ValueError('calibration candidate differs from authorized manifest')
    identity_before = _identity(candidate, policy)
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.apis import init_model

    config = Config.fromfile(authorized.config_path)
    loader_config = dict(config.train_dataloader)
    loader_config.update(batch_size=1, num_workers=0, persistent_workers=False)
    loader_config['sampler'] = dict(type='DefaultSampler', shuffle=False)
    root_determinism = seed_deterministic_root(candidate.seed)
    model = init_model(
        str(authorized.config_path),
        str(authorized.checkpoint_path),
        device=device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        model.requires_grad_(False)
    targets = discover_calibration_targets(model)
    dataloader = Runner.build_dataloader(loader_config, seed=candidate.seed,
                                         diff_rank_seed=False)
    order = hashlib.sha256()
    observed = 0
    target_config = Config.fromfile(policy)
    target_numeric = target_config.numeric_optimization
    target_features = getattr(target, 'features', {})
    raw_pwl_policy = (
        target_numeric.get('pwl')
        if target_features.get('numeric_kind') == 'pwl' else None)
    pwl_policy = ({
        name: raw_pwl_policy[name] for name in (
            'enabled_function', 'source', 'roles', 'domain', 'segments',
            'grid_points', 'saturation', 'qat_form', 'selection_policy')}
        if raw_pwl_policy is not None else None)
    session_context = (
        _HookSession(model, targets, pwl_policy=pwl_policy)
        if pwl_policy is not None else _HookSession(model, targets))
    with session_context as session, torch.inference_mode():
        for batch in dataloader:
            identifiers = _sample_ids(batch)
            for identifier in identifiers:
                order.update(identifier.encode('utf-8'))
                order.update(b'\0')
            model.test_step(batch)
            observed += len(identifiers)
            if observed >= samples:
                break
        if observed != samples:
            raise ValueError(
                f'calibration observed {observed} samples, expected {samples}')
        records = session.records()
        pwl_fit = (
            session.pwl_report(candidate_id=target.id)
            if pwl_policy is not None else None)
    policy_value = target_numeric.get('quant_policy', {})
    activation_observers = policy_value.get('activation_observers', {})
    activation_scales = _w8a8_activation_scales_from_records(
        activation_observers, records)
    identity_after = _identity(candidate, policy)
    if identity_after != identity_before:
        raise ValueError('calibration inputs changed during production run')
    artifact = {
        'schema_version': int(
            target_numeric.get('calibration', {}).get(
                'artifact_schema_version', 2)),
        'candidate_id': target.id,
        'stage': 'calibrate',
        'source': build_numeric_source_binding(
            repository_root=REPOSITORY_ROOT, candidate=target,
            manifest_path=manifest,
            policy_path=policy),
        'identity': identity_after,
        'protocol': {
            'model_mode': 'eval', 'grad_enabled': False, 'shuffle': False,
            'worker_count': 0, 'sample_count': observed,
            'sample_order_sha256': order.hexdigest(),
            'root_determinism': root_determinism,
        },
        'hooks': {
            'records': records,
            'required_records': list(_required_records(targets)),
            'unsupported_internals': list(targets.unsupported_internals),
            'activation_scales': activation_scales,
        },
    }
    if pwl_fit is not None:
        artifact['pwl_fit'] = pwl_fit
    validate_calibration_artifact(artifact)
    return artifact


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--manifest', type=Path,
                        default=REPOSITORY_ROOT / 'optimization/candidates.json')
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=str)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    try:
        policy = args.policy.resolve()
        policy.relative_to(REPOSITORY_ROOT.resolve())
        target = _candidate(args.manifest, args.candidate)
        source = (
            _candidate(args.manifest, 'full-s-v1')
            if target.route == 'ssm-quant-pwl' else target)
        if args.audit_only:
            value = audit(
                source, policy, target_candidate=target,
                manifest_path=args.manifest)
            value['target_candidate_id'] = target.id
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
        if not args.output:
            raise ValueError('--output is required for production calibration')
        output = optimization_output_path(
            args.output, repository_root=REPOSITORY_ROOT)
        _atomic_json(output, calibrate(
            source, policy, samples=args.samples, device=args.device,
            target_candidate=target, manifest_path=args.manifest))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
