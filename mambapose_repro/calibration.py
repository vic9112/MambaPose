"""Measured micro-batch admission and resolved-config materialization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from mmengine.config import Config

from .manifest import load_manifest


def resolve_batch(effective_batch: int, largest_stable_batch: int) -> tuple[int, int]:
    """Choose the largest admitted divisor and exact accumulation count."""
    if effective_batch < 1 or largest_stable_batch < 1:
        raise ValueError('batch sizes must be positive')
    limit = min(effective_batch, largest_stable_batch)
    micro_batch = max(
        candidate for candidate in range(1, limit + 1)
        if effective_batch % candidate == 0)
    return micro_batch, effective_batch // micro_batch


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain an object')
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_resolved_configs(
        manifest_path: Path | str,
        calibration_path: Path | str,
        output_dir: Path | str) -> dict[str, Any]:
    """Write one immutable runtime config for every manifest stage."""
    manifest_path = Path(manifest_path).resolve()
    repository = manifest_path.parent.parent
    calibration_path = Path(calibration_path).resolve()
    output_dir = Path(output_dir).resolve()
    calibration = _load_json(calibration_path)
    if calibration.get('schema_version') != 1:
        raise ValueError('unsupported calibration schema')
    if calibration.get('dtype') != 'fp32':
        raise ValueError('formal paper configs require fp32 calibration')
    measurements = calibration.get('runs')
    if not isinstance(measurements, dict):
        raise ValueError('calibration runs must be an object')

    entries = []
    for spec in load_manifest(manifest_path).runs:
        source = (repository / spec.config).resolve()
        source_config = Config.fromfile(source)
        destination = output_dir / f'{spec.id}.py'
        relative_base = os.path.relpath(source, destination.parent)
        lines = [
            '# Generated from measured RTX 5090 calibration; do not edit.\n',
            f'_base_ = [{relative_base!r}]\n\n',
        ]
        entry: dict[str, Any] = {
            'id': spec.id,
            'kind': spec.kind,
            'source_config': str(source),
            'path': str(destination),
        }
        if spec.kind == 'train':
            measurement = measurements.get(spec.id)
            if not isinstance(measurement, dict):
                raise ValueError(f'missing calibration for {spec.id}')
            largest = measurement.get('largest_stable_batch')
            if not isinstance(largest, int):
                raise ValueError(
                    f'{spec.id} largest_stable_batch must be an integer')
            effective = int(source_config.train_dataloader.batch_size)
            micro, accumulation = resolve_batch(effective, largest)
            lines.extend([
                f'train_dataloader = dict(batch_size={micro})\n',
                'optim_wrapper = dict('
                f'accumulative_counts={accumulation})\n',
                'reproduction_resolution = dict(\n'
                "    dtype='fp32',\n"
                f'    effective_batch={effective},\n'
                f'    micro_batch={micro},\n'
                f'    accumulation={accumulation},\n'
                f'    largest_stable_batch={largest})\n',
            ])
            entry.update({
                'effective_batch': effective,
                'micro_batch': micro,
                'accumulation': accumulation,
                'largest_stable_batch': largest,
            })
        else:
            lines.append(
                "reproduction_resolution = dict(dtype='fp32', "
                "kind='submission_export')\n")
        _atomic_text(destination, ''.join(lines))
        entries.append(entry)
    return {
        'schema_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'manifest': str(manifest_path),
        'calibration': str(calibration_path),
        'configs': entries,
    }
