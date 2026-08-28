#!/usr/bin/env python3
"""Deterministically convert or pack an admitted Route 3 weight-only model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

import torch
from mmengine.config import Config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.checkpoints import (
    authorize_manifest_candidate, authorize_tracked_config,
    build_manifest_authorized_model)
from mambapose_opt.numeric_conversion import (
    bind_numeric_inputs, install_pwl_fit, quant_policy_from_config,
    verify_numeric_inputs)
from mambapose_opt.numeric_calibration import validate_calibration_provenance
from mambapose_opt.numeric_source import build_numeric_source_binding
from mambapose_opt.pwl_artifacts import (
    build_pwl_installation_manifest, validate_pwl_fit_report,
    validate_pwl_installation_manifest)
from mambapose_opt.pwl_selection import load_pwl_selection_reference
from mambapose_opt.schema import load_candidate_manifest
from mmpose.models.utils.hardware_friendly import (
    convert_for_fake_quant, export_int8_state)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_input(path: Path, *, label: str) -> Path:
    """Keep lexical path authority so a symlink alias cannot be normalized."""
    root = REPOSITORY_ROOT.resolve(strict=True)
    supplied = Path(path)
    if any(part in {'.', '..'} for part in supplied.parts):
        raise ValueError(f'{label} path is unsafe')
    lexical = supplied if supplied.is_absolute() else root / supplied
    lexical = lexical.absolute()
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f'{label} path escapes repository') from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'{label} path must not use symlink')
    if not cursor.is_file():
        raise ValueError(f'{label} is missing')
    return cursor


def _candidate(path: Path, identifier: str):
    selected = tuple(
        item for item in load_candidate_manifest(path) if item.id == identifier)
    if len(selected) != 1:
        raise ValueError(f'candidate must resolve exactly once: {identifier}')
    return selected[0]


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


def convert(
        candidate, *, stage: str, output: Path, manifest_path: Path,
        calibration_artifact: Path | None = None,
        selection_artifact: Path | None = None) -> dict:
    kind = candidate.features.get('numeric_kind')
    if candidate.route != 'ssm-quant-pwl' \
            or kind not in {'weight-only', 'w8a8', 'pwl'}:
        raise ValueError(
            'deterministic Stage A conversion admits W8/W8A8/PWL only')
    if kind == 'pwl' and calibration_artifact is None:
        raise ValueError('PWL conversion requires completed calibration fit artifact')
    if kind == 'pwl' and selection_artifact is None:
        raise ValueError(
            'PWL conversion requires hash-bound four-candidate selection')
    if kind == 'pwl' and stage != 'convert':
        raise ValueError('PWL conversion has one install stage and no packed export')
    authorized = authorize_manifest_candidate(
        REPOSITORY_ROOT, manifest_path, candidate.id)
    if authorized.candidate != candidate:
        raise ValueError('conversion candidate differs from authorized manifest')
    commit = authorized.source['git_commit']
    config_path = authorized.config_path
    source = build_numeric_source_binding(
        repository_root=REPOSITORY_ROOT, candidate=candidate,
        manifest_path=manifest_path, policy_path=config_path,
        git_commit=commit)
    checkpoint_path = authorized.checkpoint_path
    config_authority = (
        authorize_tracked_config(
            REPOSITORY_ROOT, manifest_path, candidate)
        if kind == 'pwl' else None)
    config = (
        config_authority.load_config() if config_authority is not None
        else Config.fromfile(config_path))
    calibration = None
    selection_reference = None
    if kind == 'pwl':
        selection_artifact = _strict_input(
            selection_artifact, label='PWL selection artifact')
        selection_reference = {
            'path': selection_artifact.relative_to(
                REPOSITORY_ROOT).as_posix(),
            'sha256': _sha256(selection_artifact),
        }
        selection = load_pwl_selection_reference(
            selection_reference, repository_root=REPOSITORY_ROOT,
            manifest_path=manifest_path)
        if (selection.get('decision') != 'selected'
                or selection.get('selected_candidate_id') != candidate.id):
            raise ValueError(
                'PWL candidate is not admitted by four-candidate selection')
    if kind in {'w8a8', 'pwl'}:
        if calibration_artifact is None:
            raise ValueError(
                f'{kind} conversion requires --calibration-artifact')
        calibration_artifact = _strict_input(
            calibration_artifact, label=f'{kind} calibration artifact')
        calibration = json.loads(calibration_artifact.read_text(encoding='utf-8'))
        validate_calibration_provenance(
            calibration, expected_candidate=candidate,
            repository_root=REPOSITORY_ROOT, manifest_path=manifest_path)
        calibration_reference = {
            'path': calibration_artifact.relative_to(
                REPOSITORY_ROOT).as_posix(),
            'sha256': _sha256(calibration_artifact),
        }
        if kind == 'w8a8':
            config.numeric_optimization.quant_policy.calibration_artifact = (
                calibration_reference)
    policy = (
        quant_policy_from_config(
            config.numeric_optimization.quant_policy,
            calibration_artifact=calibration)
        if kind != 'pwl' else None)
    runtime_paths = {
        'config': config_path,
        'checkpoint': checkpoint_path,
        'policy': config_path,
    }
    if calibration_artifact is not None:
        runtime_paths['calibration'] = calibration_artifact
    if selection_artifact is not None:
        runtime_paths['selection'] = selection_artifact
    binding = bind_numeric_inputs(runtime_paths)
    verify_numeric_inputs(binding)

    if kind == 'pwl':
        model = build_manifest_authorized_model(
            REPOSITORY_ROOT, manifest_path, candidate,
            config_authority=config_authority, device='cpu')
    else:
        from mmpose.apis import init_model
        model = init_model(str(config_path), str(checkpoint_path), device='cpu')
    if kind == 'pwl':
        pwl_policy = config.numeric_optimization.pwl
        if pwl_policy.get('candidate_id') != candidate.id:
            raise ValueError('PWL policy candidate identity is invalid')
        fit = validate_pwl_fit_report(
            calibration.get('pwl_fit'), expected_candidate_id=candidate.id,
            expected_policy={
                name: pwl_policy[name] for name in (
                    'enabled_function', 'source', 'roles', 'domain',
                    'segments', 'grid_points', 'saturation', 'qat_form',
                    'selection_policy')})
        installation = build_pwl_installation_manifest(
            candidate_id=candidate.id, fit=fit,
            fit_reference=calibration_reference)
        installation_path = output.parent / 'pwl-installation.json'
        _atomic_json(installation_path, installation)
        installation_reference = {
            'path': installation_path.relative_to(REPOSITORY_ROOT).as_posix(),
            'sha256': _sha256(installation_path),
        }
        report = validate_pwl_installation_manifest(
            installation, expected_candidate_id=candidate.id,
            expected_fit_reference=calibration_reference)['report_object']
        install_pwl_fit(model, fit=fit, expected_report=report)
        config.numeric_optimization.pwl.fit_artifact = calibration_reference
        config.numeric_optimization.pwl.installation_manifest = (
            installation_reference)
        config.numeric_optimization.pwl.selection_artifact = (
            selection_reference)
    else:
        report = convert_for_fake_quant(model, policy)
    verify_numeric_inputs(binding)
    if kind == 'pwl':
        runtime_config = output.parent / 'resolved-runtime.py'
        config.dump(runtime_config)
        return {
            'schema_version': 1,
            'candidate_id': candidate.id,
            'stage': stage,
            'result': {
                'source': source,
                'runtime_bindings': {
                    role: {
                        'path': (
                            candidate.checkpoint.as_posix()
                            if role == 'checkpoint'
                            else calibration_reference['path']
                            if role == 'calibration'
                            else selection_reference['path']
                            if role == 'selection'
                            else candidate.config.as_posix()),
                        'sha256': checksum,
                    }
                    for role, checksum in binding.sha256
                },
                'runtime_config': {
                    'path': runtime_config.relative_to(
                        REPOSITORY_ROOT).as_posix(),
                    'sha256': _sha256(runtime_config),
                },
                'installation': installation_reference,
                'selection': selection_reference,
                'operation_manifest': installation['operation_manifest'],
                'latency_claim': (
                    'none-pwl-pytorch-runtime-is-not-fpga-proof'),
            },
        }
    result = {
        'source': source,
        'runtime_bindings': {
            role: {
                'path': (
                    candidate.checkpoint.as_posix() if role == 'checkpoint'
                    else calibration_artifact.relative_to(
                        REPOSITORY_ROOT).as_posix()
                    if role == 'calibration'
                    else candidate.config.as_posix()),
                'sha256': checksum,
            }
            for role, checksum in binding.sha256
        },
        'conversion': asdict(report),
        'precision_invariants': dict(
            config.numeric_optimization.precision_invariants),
        'latency_claim': 'none-fake-quant-is-not-an-integer-kernel',
    }
    if calibration_artifact is not None:
        runtime_config = output.parent / 'resolved-runtime.py'
        config.dump(runtime_config)
        result['runtime_config'] = {
            'path': runtime_config.relative_to(REPOSITORY_ROOT).as_posix(),
            'sha256': _sha256(runtime_config),
        }
    if stage == 'export':
        packed_path = output.with_suffix('.int8.pt')
        packed = {
            'schema_version': 1,
            'candidate_id': candidate.id,
            'weights': export_int8_state(model, report),
        }
        torch.save(packed, packed_path, _use_new_zipfile_serialization=False)
        result['export'] = {
            'path': packed_path.resolve().relative_to(
                REPOSITORY_ROOT.resolve()).as_posix(),
            'sha256': _sha256(packed_path),
            'bytes': packed_path.stat().st_size,
            'format': 'symmetric-int8-per-output-channel-v1',
        }
    return {
        'schema_version': 1,
        'candidate_id': candidate.id,
        'stage': stage,
        'result': result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_id')
    parser.add_argument('--stage', choices=('convert', 'export'), required=True)
    parser.add_argument('--manifest', type=Path,
                        default=REPOSITORY_ROOT / 'optimization/candidates.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration-artifact', type=Path)
    parser.add_argument('--selection-artifact', type=Path)
    args = parser.parse_args()
    try:
        output = optimization_output_path(
            args.output, repository_root=REPOSITORY_ROOT)
        candidate = _candidate(args.manifest, args.candidate_id)
        _atomic_json(output, convert(
            candidate, stage=args.stage, output=output,
            manifest_path=args.manifest,
            calibration_artifact=args.calibration_artifact,
            selection_artifact=args.selection_artifact))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
