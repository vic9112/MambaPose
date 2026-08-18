"""Sequential real-data, resume, and measured-batch campaign gates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from mmengine.config import Config

from .checkpoint import validate_checkpoint
from .manifest import Manifest, RunSpec, load_manifest


class GateError(RuntimeError):
    pass


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class GateRunner:

    def __init__(self, repository: Path | str):
        self.repository = Path(repository).resolve()
        self.campaign = self.repository / 'work_dirs/reproduction'
        self.manifest = load_manifest(
            self.repository / 'reproduction/manifest.json')
        self.python = self.repository / '.venv/bin/python'
        self.worker = self.repository / 'tools/reproduction/gate_worker.py'

    def _config(self, spec: RunSpec, *, resolved: bool = False) -> Path:
        if resolved:
            candidate = (
                self.campaign / 'resolved_configs' / f'{spec.id}.py')
            if not candidate.is_file():
                raise GateError(f'missing resolved config for {spec.id}')
            return candidate
        return self.repository / spec.config

    def _worker(
            self, *, label: str, mode: str, config: Path, batch_size: int,
            work_dir: Path, output: Path, epochs: int = 1,
            save_checkpoint: bool = False, resume: Path | None = None,
            checkpoint: Path | None = None) -> tuple[int, dict[str, Any]]:
        command = [
            str(self.python), str(self.worker),
            '--mode', mode,
            '--config', str(config),
            '--work-dir', str(work_dir),
            '--output', str(output),
            '--batch-size', str(batch_size),
            '--epochs', str(epochs),
        ]
        if save_checkpoint:
            command.append('--save-checkpoint')
        if resume is not None:
            command.extend(['--resume', str(resume)])
        if checkpoint is not None:
            command.extend(['--checkpoint', str(checkpoint)])
        log = output.with_suffix('.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update({
            'PYTHONNOUSERSITE': '1',
            'CUDA_VISIBLE_DEVICES': '0',
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1',
        })
        with log.open('a', encoding='utf-8') as stream:
            stream.write(f'command={json.dumps(command)}\n')
            stream.flush()
            result = subprocess.run(
                command, cwd=self.repository, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
        evidence = {}
        if output.is_file():
            evidence = json.loads(output.read_text(encoding='utf-8'))
        if result.returncode not in {0, 42}:
            raise GateError(
                f'{label} failed with exit {result.returncode}; see {log}')
        return result.returncode, evidence

    @staticmethod
    def _primary_specs(manifest: Manifest) -> tuple[RunSpec, ...]:
        ids = {
            'coco-s-v1', 'coco-s-v2', 'coco-b',
            'crowdpose-s-v1', 'crowdpose-s-v2',
        }
        return tuple(spec for spec in manifest.runs if spec.id in ids)

    def real_data_smoke(self) -> dict[str, Any]:
        root = self.campaign / 'evidence/smoke/real-data'
        entries = []
        for spec in self._primary_specs(self.manifest):
            run_root = root / spec.id
            train_output = run_root / 'train.json'
            code, train = self._worker(
                label=f'{spec.id} real train smoke',
                mode='train', config=self._config(spec), batch_size=1,
                work_dir=run_root / 'train', output=train_output,
                save_checkpoint=True)
            if code or train.get('status') != 'passed':
                raise GateError(f'{spec.id} real train smoke did not pass')
            checkpoint = Path(train['checkpoint'])
            validation = validate_checkpoint(checkpoint)
            if not validation.valid:
                raise GateError(
                    f'{spec.id} smoke checkpoint is invalid: '
                    f'{validation.error}')
            code, test = self._worker(
                label=f'{spec.id} real test smoke',
                mode='test', config=self._config(spec), batch_size=1,
                work_dir=run_root / 'test', output=run_root / 'test.json',
                checkpoint=checkpoint)
            if code or test.get('status') != 'passed':
                raise GateError(f'{spec.id} real test smoke did not pass')
            entries.append({'id': spec.id, 'train': train, 'test': test})
        report = {
            'schema_version': 1,
            'gate': 'real-data-smoke',
            'verified_at': datetime.now(timezone.utc).isoformat(),
            'runs': entries,
        }
        _atomic_json(root / 'summary.json', report)
        return report

    def resume_smoke(self) -> dict[str, Any]:
        spec = next(
            item for item in self.manifest.runs if item.id == 'coco-s-v1')
        root = self.campaign / 'evidence/smoke/resume'
        first_output = root / 'epoch-1.json'
        code, first = self._worker(
            label='resume smoke checkpoint creation',
            mode='train', config=self._config(spec), batch_size=1,
            work_dir=root / 'run', output=first_output,
            epochs=1, save_checkpoint=True)
        if code or first.get('status') != 'passed':
            raise GateError('resume smoke did not produce the first checkpoint')
        first_checkpoint = Path(first['checkpoint'])
        if not validate_checkpoint(
                first_checkpoint, require_training_state=True).valid:
            raise GateError('resume smoke first checkpoint is invalid')
        code, second = self._worker(
            label='resume smoke restarted process',
            mode='train', config=self._config(spec), batch_size=1,
            work_dir=root / 'run', output=root / 'epoch-2.json',
            epochs=2, save_checkpoint=True, resume=first_checkpoint)
        if code or second.get('status') != 'passed':
            raise GateError('resume smoke restarted process did not pass')
        second_checkpoint = Path(second['checkpoint'])
        if (second_checkpoint == first_checkpoint
                or not validate_checkpoint(second_checkpoint).valid):
            raise GateError('resume smoke did not advance to a valid checkpoint')
        report = {
            'schema_version': 1,
            'gate': 'resume-smoke',
            'verified_at': datetime.now(timezone.utc).isoformat(),
            'first': first,
            'resumed': second,
            'process_boundary_proven': True,
        }
        _atomic_json(root / 'summary.json', report)
        return report

    def calibrate(self, *, maximum_reserved_fraction: float = 0.90) -> dict[str, Any]:
        root = self.campaign / 'evidence/calibration'
        measurements: dict[str, dict[str, Any]] = {}
        for spec in self._primary_specs(self.manifest):
            config = self._config(spec)
            effective = int(Config.fromfile(config).train_dataloader.batch_size)
            candidates = [
                value for value in range(1, effective + 1)
                if effective % value == 0 and value & (value - 1) == 0
            ]
            largest = 0
            attempts = []
            for batch in candidates:
                attempt_root = root / spec.id / f'batch-{batch}'
                code, evidence = self._worker(
                    label=f'{spec.id} batch calibration {batch}',
                    mode='train', config=config, batch_size=batch,
                    work_dir=attempt_root / 'run',
                    output=attempt_root / 'result.json')
                entry = {'batch_size': batch, 'exit_code': code, **evidence}
                attempts.append(entry)
                if code == 42:
                    break
                reserved_fraction = (
                    evidence['peak_reserved_bytes']
                    / evidence['device_total_bytes'])
                entry['reserved_fraction'] = reserved_fraction
                if reserved_fraction > maximum_reserved_fraction:
                    entry['status'] = 'headroom_rejected'
                    break
                largest = batch
            if largest < 1:
                raise GateError(f'{spec.id} has no admitted FP32 batch')
            measurements[spec.id] = {
                'effective_batch': effective,
                'largest_stable_batch': largest,
                'maximum_reserved_fraction': maximum_reserved_fraction,
                'attempts': attempts,
            }
        inherited = {
            'coco-s-v1-no-pif': 'coco-s-v1',
            'crowdpose-s-v1-no-pif': 'crowdpose-s-v1',
            'crowdpose-s-v1-no-prior': 'crowdpose-s-v1',
            'crowdpose-s-v1-no-cycling': 'crowdpose-s-v1',
        }
        for run_id, parent in inherited.items():
            measurements[run_id] = {
                'effective_batch': measurements[parent]['effective_batch'],
                'largest_stable_batch': measurements[parent][
                    'largest_stable_batch'],
                'inherited_conservatively_from': parent,
            }
        report = {
            'schema_version': 1,
            'dtype': 'fp32',
            'verified_at': datetime.now(timezone.utc).isoformat(),
            'gpu_policy': {
                'maximum_reserved_fraction': maximum_reserved_fraction,
                'tf32': False,
                'amp': False,
            },
            'runs': measurements,
        }
        _atomic_json(self.campaign / 'evidence/calibration.json', report)
        return report

    def resolved_smoke(self) -> dict[str, Any]:
        root = self.campaign / 'evidence/smoke/resolved'
        entries = []
        for spec in self.manifest.runs:
            config = self._config(spec, resolved=True)
            loaded = Config.fromfile(config)
            if spec.kind == 'export':
                entries.append({'id': spec.id, 'status': 'config-loaded'})
                continue
            code, evidence = self._worker(
                label=f'{spec.id} resolved train smoke',
                mode='train', config=config, batch_size=1,
                work_dir=root / spec.id / 'run',
                output=root / spec.id / 'result.json')
            if code or evidence.get('status') != 'passed':
                raise GateError(f'{spec.id} resolved smoke did not pass')
            entries.append({
                'id': spec.id,
                'status': 'passed',
                'effective_batch': loaded.reproduction_resolution.effective_batch,
                'micro_batch': loaded.reproduction_resolution.micro_batch,
                'accumulation': loaded.reproduction_resolution.accumulation,
                'evidence': evidence,
            })
        report = {
            'schema_version': 1,
            'gate': 'resolved-smoke',
            'verified_at': datetime.now(timezone.utc).isoformat(),
            'runs': entries,
        }
        _atomic_json(root / 'summary.json', report)
        return report
