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
from mambapose_opt.numeric_conversion import (
    bind_numeric_inputs, quant_policy_from_config, verify_numeric_inputs)
from mambapose_opt.numeric_calibration import validate_calibration_artifact
from mambapose_opt.numeric_source import build_numeric_source_binding
from mambapose_opt.source import clean_git_commit
from mambapose_opt.schema import load_candidate_manifest
from mmpose.models.utils.hardware_friendly import (
    convert_for_fake_quant, export_int8_state)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
        calibration_artifact: Path | None = None) -> dict:
    if candidate.route != 'ssm-quant-pwl' \
            or candidate.features.get('numeric_kind') not in {
                'weight-only', 'w8a8'}:
        raise ValueError(
            'deterministic Stage A conversion admits W8/W8A8 only')
    commit = clean_git_commit(REPOSITORY_ROOT)
    config_path = (REPOSITORY_ROOT / candidate.config).resolve()
    checkpoint_path = (REPOSITORY_ROOT / candidate.checkpoint).resolve()
    config = Config.fromfile(config_path)
    calibration = None
    if candidate.features.get('numeric_kind') == 'w8a8':
        if calibration_artifact is None:
            raise ValueError('W8A8 conversion requires --calibration-artifact')
        calibration_artifact = calibration_artifact.resolve()
        calibration_artifact.relative_to(REPOSITORY_ROOT.resolve())
        calibration = json.loads(calibration_artifact.read_text(encoding='utf-8'))
        validate_calibration_artifact(
            calibration, expected_candidate_id=candidate.id)
        config.numeric_optimization.quant_policy.calibration_artifact = {
            'path': calibration_artifact.relative_to(
                REPOSITORY_ROOT).as_posix(),
            'sha256': _sha256(calibration_artifact),
        }
    policy = quant_policy_from_config(
        config.numeric_optimization.quant_policy,
        calibration_artifact=calibration)
    runtime_paths = {
        'config': config_path,
        'checkpoint': checkpoint_path,
        'policy': config_path,
    }
    if calibration_artifact is not None:
        runtime_paths['calibration'] = calibration_artifact
    binding = bind_numeric_inputs(runtime_paths)
    verify_numeric_inputs(binding)

    from mmpose.apis import init_model
    model = init_model(str(config_path), str(checkpoint_path), device='cpu')
    report = convert_for_fake_quant(model, policy)
    verify_numeric_inputs(binding)
    result = {
        'source': build_numeric_source_binding(
            repository_root=REPOSITORY_ROOT, candidate=candidate,
            manifest_path=manifest_path, policy_path=config_path,
            git_commit=commit),
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
    args = parser.parse_args()
    try:
        output = optimization_output_path(
            args.output, repository_root=REPOSITORY_ROOT)
        candidate = _candidate(args.manifest, args.candidate_id)
        _atomic_json(output, convert(
            candidate, stage=args.stage, output=output,
            manifest_path=args.manifest,
            calibration_artifact=args.calibration_artifact))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
