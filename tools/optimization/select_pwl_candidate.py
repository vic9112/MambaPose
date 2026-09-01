#!/usr/bin/env python3
"""Select one PWL candidate from exactly four calibrated schema-v3 fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mambapose_opt.artifacts import optimization_output_path
from mambapose_opt.pwl_selection import (
    CANONICAL_PWL_CANDIDATES, build_pwl_selection_artifact,
    validate_pwl_selection_artifact)
from mambapose_opt.pwl_paths import canonical_path, canonical_relative_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
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


def _calibrations(values: list[str]) -> dict[str, dict[str, str]]:
    expected = tuple(item[0] for item in CANONICAL_PWL_CANDIDATES)
    if len(values) != len(expected):
        raise ValueError('selection requires exactly four --calibration rows')
    result = {}
    for value in values:
        candidate_id, separator, raw_path = value.partition('=')
        if not separator or candidate_id not in expected or candidate_id in result:
            raise ValueError(
                '--calibration must be one unique canonical candidate=path')
        supplied = canonical_relative_path(
            raw_path, label=f'{candidate_id} calibration')
        if not supplied.is_absolute():
            supplied = REPOSITORY_ROOT / supplied
        path = supplied.resolve(strict=True)
        relative = path.relative_to(REPOSITORY_ROOT.resolve())
        result[candidate_id] = {
            'path': relative.as_posix(), 'sha256': _sha256(path)}
    if tuple(result) != expected:
        raise ValueError(
            '--calibration rows must use canonical SiLU/GELU/softplus/exp order')
    return result


def _manifest_path(value: str) -> Path:
    try:
        return canonical_path(
            value, label='PWL source manifest', allow_absolute=True)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _selection_output(value: str) -> str:
    try:
        return canonical_relative_path(
            value, label='PWL selection output').as_posix()
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=_manifest_path,
        default=REPOSITORY_ROOT / 'optimization/candidates.json')
    parser.add_argument(
        '--calibration', action='append', default=[],
        help='canonical-candidate-id=repository-relative-calibrate.json')
    parser.add_argument(
        '--output', type=_selection_output, default=(
            'work_dirs/optimization/ssm-quant-pwl/'
            'pwl-selection/selection.json'))
    args = parser.parse_args()
    try:
        output = optimization_output_path(
            args.output, repository_root=REPOSITORY_ROOT)
        expected_output = (
            REPOSITORY_ROOT / 'work_dirs/optimization/ssm-quant-pwl/'
            'pwl-selection/selection.json').resolve()
        if output.resolve() != expected_output:
            raise ValueError('PWL selection output path is not canonical')
        artifact = build_pwl_selection_artifact(
            repository_root=REPOSITORY_ROOT,
            manifest_path=args.manifest,
            calibration_references=_calibrations(args.calibration))
        _atomic_json(output, artifact)
        validate_pwl_selection_artifact(
            json.loads(output.read_text(encoding='utf-8')),
            repository_root=REPOSITORY_ROOT,
            manifest_path=args.manifest)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
