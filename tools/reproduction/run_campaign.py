#!/usr/bin/env python3
"""Run the manifest sequentially with validated checkpoints and finite retry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mambapose_repro.checkpoint import (
    PermanentCheckpointError, select_evaluation_checkpoint, select_resume,
    validate_checkpoint)
from mambapose_repro.manifest import RunSpec, load_manifest
from mambapose_repro.orchestrator import (
    AttemptOutcome, CampaignLock, PERMANENT_EXIT, PermanentFailure,
    RetryExhausted, TRANSIENT_EXIT, failure_fingerprint,
    run_until_terminal)
from mambapose_repro.calibration import materialize_resolved_configs
from mambapose_repro.gates import GateError, GateRunner
from mambapose_repro.state import StateStore


CAMPAIGN_DIR = REPO_ROOT / 'work_dirs/reproduction'
MANIFEST_PATH = REPO_ROOT / 'reproduction/manifest.json'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise PermanentFailure(f'required provenance file is missing: {path}')
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _repo_commit() -> str:
    dirty = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    if dirty.stdout.strip():
        raise PermanentFailure(
            'formal campaign requires a clean tracked Git worktree')
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True)
    return result.stdout.strip()


def _provenance(config_path: Path) -> dict[str, str]:
    return {
        'config_sha256': _sha256(config_path),
        'repo_commit': _repo_commit(),
        'data_inventory_sha256': _sha256(REPO_ROOT / 'data/inventory.json'),
        'environment_sha256': _combined_hash([
            REPO_ROOT / 'work_dirs/reproduction/evidence/environment.json',
            REPO_ROOT / 'work_dirs/reproduction/evidence/native-build.json',
            REPO_ROOT / 'work_dirs/reproduction/evidence/native.json',
            REPO_ROOT / 'work_dirs/reproduction/evidence/rebuild.json',
            REPO_ROOT / 'work_dirs/reproduction/evidence/pip-freeze.txt',
        ]),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _admit_provenance(work_dir: Path, expected: dict[str, str]) -> None:
    path = work_dir / 'provenance.json'
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        if actual != expected:
            raise PermanentFailure(
                f'work directory provenance mismatch at {path}')
    else:
        _atomic_json(path, expected)


def _tail(path: Path, lines: int = 120) -> str:
    try:
        return '\n'.join(path.read_text(
            encoding='utf-8', errors='replace').splitlines()[-lines:])
    except OSError:
        return ''


def _classify_exit(returncode: int, log_tail: str) -> AttemptOutcome:
    if returncode == 0:
        return AttemptOutcome(0, 'process-exit-0')
    lowered = log_tail.lower()
    transient_markers = (
        'nccl remote error', 'connection reset', 'temporarily unavailable',
        'input/output error', 'no space left on device')
    permanent_markers = (
        'cuda out of memory', 'filenotfounderror', 'no such file or directory',
        'keyerror', 'assertionerror', 'runtimeerror: error compiling',
        'valueerror')
    fingerprint = failure_fingerprint(log_tail or f'exit={returncode}')
    if returncode < 0 or any(marker in lowered for marker in transient_markers):
        return AttemptOutcome(TRANSIENT_EXIT, fingerprint)
    if any(marker in lowered for marker in permanent_markers):
        return AttemptOutcome(PERMANENT_EXIT, fingerprint)
    return AttemptOutcome(PERMANENT_EXIT, fingerprint)


def _run_command(
        run_id: str,
        phase: str,
        command: list[str],
        work_dir: Path,
        *,
        stall_seconds: int = 7200) -> AttemptOutcome:
    log_path = work_dir / f'{phase}.log'
    heartbeat = CAMPAIGN_DIR / 'heartbeat.json'
    environment = os.environ.copy()
    environment.update({
        'PYTHONNOUSERSITE': '1',
        'CUDA_VISIBLE_DEVICES': '0',
    })
    with log_path.open('a', encoding='utf-8') as log:
        log.write(
            f'\n[{datetime.now(timezone.utc).isoformat()}] '
            f'command={json.dumps(command)}\n')
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True)
        last_activity = time.monotonic()
        last_size = log_path.stat().st_size
        while True:
            returncode = process.poll()
            size = log_path.stat().st_size
            if size != last_size:
                last_activity = time.monotonic()
                last_size = size
            _atomic_json(heartbeat, {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'run_id': run_id,
                'phase': phase,
                'pid': process.pid,
                'log_path': str(log_path.relative_to(REPO_ROOT)),
                'log_bytes': size,
            })
            if returncode is not None:
                break
            if time.monotonic() - last_activity > stall_seconds:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return AttemptOutcome(
                    TRANSIENT_EXIT,
                    failure_fingerprint(
                        f'stalled:{run_id}:{phase}:{_tail(log_path)}'))
            time.sleep(30)
    return _classify_exit(process.returncode, _tail(log_path))


def _valid_finite_json(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False

    def finite(item: Any) -> bool:
        if isinstance(item, dict):
            return all(finite(child) for child in item.values())
        if isinstance(item, list):
            return all(finite(child) for child in item)
        if isinstance(item, float):
            return math.isfinite(item)
        return True

    return isinstance(value, dict) and finite(value)


def _resolved_config(spec: RunSpec) -> Path:
    candidate = CAMPAIGN_DIR / 'resolved_configs' / f'{spec.id}.py'
    return candidate if candidate.is_file() else REPO_ROOT / spec.config


def _completion_valid(work_dir: Path, provenance: dict[str, str]) -> bool:
    path = work_dir / 'completion.json'
    if not path.is_file():
        return False
    try:
        completion = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return False
    if completion.get('provenance') != provenance:
        return False
    return all(
        (REPO_ROOT / artifact['path']).is_file()
        and _sha256(REPO_ROOT / artifact['path']) == artifact['sha256']
        for artifact in completion.get('artifacts', []))


class CampaignExecutor:
    def __init__(self):
        self.manifest = load_manifest(MANIFEST_PATH)
        self.by_id = {run.id: run for run in self.manifest.runs}

    def _train(self, spec: RunSpec) -> AttemptOutcome:
        work_dir = REPO_ROOT / spec.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        config = _resolved_config(spec)
        provenance = _provenance(config)
        _admit_provenance(work_dir, provenance)
        if _completion_valid(work_dir, provenance):
            return AttemptOutcome(0, 'validated-existing-completion')
        try:
            resume = select_resume(work_dir, provenance)
        except PermanentCheckpointError as error:
            return AttemptOutcome(
                PERMANENT_EXIT, failure_fingerprint(str(error)))
        command = [
            str(REPO_ROOT / '.venv/bin/python'),
            str(REPO_ROOT / 'tools/train.py'),
            str(config),
            '--work-dir', str(work_dir),
        ]
        if resume is not None:
            command.extend(['--resume', str(resume.path)])
        outcome = _run_command(spec.id, 'train', command, work_dir)
        if outcome.exit_code:
            return outcome
        try:
            checkpoint = select_evaluation_checkpoint(work_dir)
        except PermanentCheckpointError as error:
            return AttemptOutcome(
                PERMANENT_EXIT, failure_fingerprint(str(error)), False)
        metrics = work_dir / 'metrics.json'
        outcome = _run_command(spec.id, 'evaluate', [
            str(REPO_ROOT / '.venv/bin/python'),
            str(REPO_ROOT / 'tools/test.py'),
            str(config),
            str(checkpoint),
            '--work-dir', str(work_dir / 'evaluation'),
            '--out', str(metrics),
        ], work_dir)
        if outcome.exit_code:
            return outcome
        artifacts_valid = (
            validate_checkpoint(checkpoint).valid
            and _valid_finite_json(metrics))
        if artifacts_valid:
            _atomic_json(work_dir / 'completion.json', {
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'provenance': provenance,
                'artifacts': [
                    {
                        'path': str(checkpoint.relative_to(REPO_ROOT)),
                        'sha256': _sha256(checkpoint),
                    },
                    {
                        'path': str(metrics.relative_to(REPO_ROOT)),
                        'sha256': _sha256(metrics),
                    },
                ],
            })
        return AttemptOutcome(
            0, 'trained-and-evaluated', artifacts_valid=artifacts_valid)

    def _export(self, spec: RunSpec) -> AttemptOutcome:
        dependency = self.by_id[spec.depends_on[0]]
        dependency_dir = REPO_ROOT / dependency.work_dir
        work_dir = REPO_ROOT / spec.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        config = _resolved_config(spec)
        provenance = _provenance(config)
        _admit_provenance(work_dir, provenance)
        if _completion_valid(work_dir, provenance):
            return AttemptOutcome(0, 'validated-existing-completion')
        try:
            checkpoint = select_evaluation_checkpoint(dependency_dir)
        except PermanentCheckpointError as error:
            return AttemptOutcome(
                PERMANENT_EXIT, failure_fingerprint(str(error)))
        metrics = work_dir / 'submission-metrics.json'
        outcome = _run_command(spec.id, 'export', [
            str(REPO_ROOT / '.venv/bin/python'),
            str(REPO_ROOT / 'tools/test.py'),
            str(config),
            str(checkpoint),
            '--work-dir', str(work_dir),
            '--out', str(metrics),
        ], work_dir)
        if outcome.exit_code:
            return outcome
        from mmengine.config import Config
        cfg = Config.fromfile(config)
        submission = REPO_ROOT / (
            cfg.test_evaluator.outfile_prefix + '.keypoints.json')
        valid = submission.is_file() and submission.stat().st_size > 2
        if valid:
            _atomic_json(work_dir / 'completion.json', {
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'provenance': provenance,
                'artifacts': [{
                    'path': str(submission.relative_to(REPO_ROOT)),
                    'sha256': _sha256(submission),
                }],
            })
        return AttemptOutcome(0, 'submission-exported', artifacts_valid=valid)

    def run(self, spec: RunSpec) -> AttemptOutcome:
        return self._train(spec) if spec.kind == 'train' else self._export(spec)

    def completion_valid(self, spec: RunSpec) -> bool:
        config = _resolved_config(spec)
        provenance = _provenance(config)
        return _completion_valid(REPO_ROOT / spec.work_dir, provenance)


def _render_status(require_complete: bool = False) -> int:
    manifest = load_manifest(MANIFEST_PATH)
    store = StateStore(CAMPAIGN_DIR)
    state = store.read()
    runs = state.get('runs', {})
    rendered = {
        'generation': state.get('generation'),
        'stage': state.get('stage'),
        'current_run': state.get('current_run'),
        'runs': {
            spec.id: runs.get(spec.id, {'status': 'pending'})
            for spec in manifest.runs
        },
    }
    print(json.dumps(rendered, indent=2, sort_keys=True))
    complete = all(
        runs.get(spec.id, {}).get('status') == 'complete'
        for spec in manifest.runs)
    return 0 if complete or not require_complete else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--diagnose-current', action='store_true')
    action.add_argument(
        '--gate', choices=(
            'real-data-smoke', 'resume-smoke', 'calibrate',
            'resolved-smoke'))
    action.add_argument('--resolve-configs', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--dtype', choices=('fp32',), default='fp32')
    parser.add_argument('--interrupt-after-checkpoint', action='store_true')
    args = parser.parse_args()
    if args.status or args.diagnose_current:
        return _render_status(args.require_complete)
    if args.gate:
        gates = GateRunner(REPO_ROOT)
        try:
            if args.gate == 'real-data-smoke':
                result = gates.real_data_smoke()
            elif args.gate == 'resume-smoke':
                result = gates.resume_smoke()
            elif args.gate == 'calibrate':
                result = gates.calibrate()
            else:
                result = gates.resolved_smoke()
        except GateError as error:
            print(str(error), file=sys.stderr)
            return PERMANENT_EXIT
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.resolve_configs:
        report = materialize_resolved_configs(
            MANIFEST_PATH,
            CAMPAIGN_DIR / 'evidence/calibration.json',
            CAMPAIGN_DIR / 'resolved_configs')
        _atomic_json(
            CAMPAIGN_DIR / 'evidence/resolved-configs.json', report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    store = StateStore(CAMPAIGN_DIR)
    executor = CampaignExecutor()
    try:
        with CampaignLock(CAMPAIGN_DIR / 'campaign.lock'):
            for spec in executor.manifest.runs:
                run_state = store.read().get('runs', {}).get(spec.id, {})
                if run_state.get('status') == 'complete':
                    if executor.completion_valid(spec):
                        continue
                    raise PermanentFailure(
                        f'{spec.id} is recorded complete but its provenance '
                        'or artifacts no longer validate')
                dependencies_complete = all(
                    store.read().get('runs', {}).get(dependency, {}).get(
                        'status') == 'complete'
                    for dependency in spec.depends_on)
                if not dependencies_complete:
                    raise PermanentFailure(
                        f'{spec.id} dependency is not complete')
                run_until_terminal(
                    spec.id,
                    lambda spec=spec: executor.run(spec),
                    store,
                    max_attempts=spec.max_attempts)
    except (PermanentFailure, RetryExhausted) as error:
        print(str(error), file=sys.stderr)
        return PERMANENT_EXIT
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
