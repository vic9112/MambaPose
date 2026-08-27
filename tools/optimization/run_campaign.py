#!/usr/bin/env python3
"""Advance the durable optimization campaign under the common GPU lease."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mambapose_opt.controller import OptimizationController, StageOutcome
from mambapose_opt.schema import (
    CandidateManifestError,
    CandidateSpec,
    load_candidate_manifest,
)
from mambapose_repro.orchestrator import PERMANENT_EXIT, failure_fingerprint


CAMPAIGN_ROOT = REPO_ROOT / 'work_dirs/optimization'
MANIFEST_PATH = REPO_ROOT / 'optimization/candidates.json'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SubprocessStageRunner:
    """Execute route tools without a shell and return hashed artifacts."""

    def __init__(
            self, campaign_root: Path, manifest_path: Path,
            *, device_index: int = 0, heartbeat_interval: float = 30.0):
        self.campaign_root = campaign_root
        self.manifest_path = manifest_path
        self.device_index = device_index
        self.heartbeat_interval = heartbeat_interval

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({
            'PYTHONNOUSERSITE': '1',
            'CUDA_VISIBLE_DEVICES': str(self.device_index),
            'MAMBAPOSE_PHYSICAL_DEVICE_INDEX': str(self.device_index),
            'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1',
        })
        return environment

    def wait_with_heartbeat(
            self, process: subprocess.Popen, heartbeat_path: Path,
            payload: dict[str, object]) -> int:
        while True:
            returncode = process.poll()
            _atomic_json(heartbeat_path, {
                **payload,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'pid': process.pid,
                'phase': 'exited' if returncode is not None else 'running',
            })
            if returncode is not None:
                return returncode
            time.sleep(self.heartbeat_interval)

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()

    def _command(
            self, candidate: CandidateSpec, stage: str,
            artifact: Path) -> list[str]:
        python = str(REPO_ROOT / '.venv/bin/python')
        common = [candidate.id, '--output', self._relative(artifact)]
        if stage == 'profile':
            return [
                python, str(REPO_ROOT / 'tools/optimization/profile_model.py'),
                candidate.id, '--manifest', str(self.manifest_path),
                '--output', self._relative(artifact),
            ]
        tool = {
            'calibrate': 'calibrate_candidate.py',
            'train': 'train_candidate.py',
            'evaluate': 'evaluate_candidate.py',
            'latency': 'measure_latency.py',
            'compare': 'compare_candidates.py',
        }[stage]
        if stage in {'evaluate', 'latency'}:
            common = [
                candidate.id, '--manifest', str(self.manifest_path),
                '--output', self._relative(artifact),
            ]
        return [python, str(REPO_ROOT / 'tools/optimization' / tool), *common]

    def __call__(
            self, candidate: CandidateSpec, stage: str,
            stage_dir: Path, attempt: int) -> StageOutcome:
        artifact = stage_dir / f'{stage}.json'
        command = self._command(candidate, stage, artifact)
        missing_tool = Path(command[1])
        if not missing_tool.is_file():
            message = f'optimization stage tool is missing: {missing_tool}'
            return StageOutcome(
                f'{candidate.id}:{stage}', stage, candidate.id,
                PERMANENT_EXIT, failure_fingerprint(message),
                attempt=attempt, message=message)

        log_path = stage_dir / f'attempt-{attempt}.log'
        heartbeat_path = self.campaign_root / 'heartbeat.json'
        environment = self.environment()
        with log_path.open('a', encoding='utf-8') as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=False,
            )
            returncode = self.wait_with_heartbeat(process, heartbeat_path, {
                'stage_id': f'{candidate.id}:{stage}',
                'attempt': attempt,
                'log_path': self._relative(log_path),
            })

        message = log_path.read_text(
            encoding='utf-8', errors='replace')[-4096:]
        if returncode != 0:
            exit_code = returncode if returncode in {75, 78} else 78
            return StageOutcome(
                f'{candidate.id}:{stage}', stage, candidate.id,
                exit_code, failure_fingerprint(message or f'exit={returncode}'),
                attempt=attempt, message=message)
        valid = artifact.is_file()
        hashes = {str(artifact): _sha256(artifact)} if valid else {}
        return StageOutcome(
            f'{candidate.id}:{stage}', stage, candidate.id,
            0, failure_fingerprint(f'{candidate.id}:{stage}:complete'),
            artifacts_valid=valid,
            artifacts=(artifact,),
            artifact_sha256=hashes,
            attempt=attempt,
        )


def _status(root: Path) -> int:
    path = root / 'state.json'
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        value = {
            'schema_version': 1,
            'generation': 0,
            'stage': 'not_started',
            'current_run': None,
            'runs': {},
        }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _select(
        candidates: Sequence[CandidateSpec],
        identifiers: Sequence[str]) -> tuple[CandidateSpec, ...]:
    if not identifiers:
        return tuple(candidates)
    wanted = set(identifiers)
    selected = tuple(item for item in candidates if item.id in wanted)
    missing = wanted - {item.id for item in selected}
    if missing:
        raise CandidateManifestError(
            f'candidate not found: {sorted(missing)}')
    return selected


def _canonical_checkout_root() -> Path:
    result = subprocess.run(
        ['git', 'rev-parse', '--git-common-dir'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = REPO_ROOT / common
    common = common.resolve()
    if common.name != '.git':
        raise ValueError(
            f'Git common directory is not a checkout .git directory: {common}')
    return common.parent


def _canonical_gpu_lock() -> Path:
    return _canonical_checkout_root() / 'work_dirs/optimization/gpu.lock'


def _validated_gpu_lock(override: Path | None) -> tuple[Path, Path]:
    checkout_root = _canonical_checkout_root()
    canonical = (
        checkout_root / 'work_dirs/optimization/gpu.lock').resolve()
    selected = Path(override or canonical).resolve()
    if selected != canonical:
        raise ValueError(
            f'GPU lock override must equal canonical GPU lock: {canonical}')
    return checkout_root, canonical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run', action='store_true')
    action.add_argument('--status', action='store_true')
    parser.add_argument('--candidate', action='append', default=[])
    parser.add_argument('--manifest', type=Path, default=MANIFEST_PATH)
    parser.add_argument('--campaign-root', type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--gpu-lock-path', type=Path)
    args = parser.parse_args()
    if args.status:
        try:
            campaign_root = args.campaign_root.resolve()
            campaign_root.relative_to(REPO_ROOT.resolve())
            if (
                    campaign_root.name != 'optimization'
                    or campaign_root.parent.name != 'work_dirs'):
                raise ValueError(
                    'optimization campaign root must end in '
                    'work_dirs/optimization')
        except (OSError, RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return PERMANENT_EXIT
        return _status(campaign_root)

    try:
        candidates = _select(
            load_candidate_manifest(args.manifest), args.candidate)
    except CandidateManifestError as error:
        print(str(error), file=sys.stderr)
        return PERMANENT_EXIT

    try:
        shared_lock_root, gpu_lock_path = _validated_gpu_lock(
            args.gpu_lock_path)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f'cannot derive canonical GPU lock: {error}', file=sys.stderr)
        return PERMANENT_EXIT
    expected_run_ids = tuple(
        f'{candidate.id}:{stage}'
        for candidate in candidates
        for stage in ('profile', 'calibrate', 'train', 'evaluate', 'latency',
                      'compare'))
    try:
        runner = SubprocessStageRunner(
            args.campaign_root, args.manifest, device_index=args.device_index)
        for candidate in candidates:
            controller = OptimizationController(
                args.campaign_root,
                candidate,
                runner,
                repository_root=REPO_ROOT,
                manifest_path=args.manifest,
                device_index=args.device_index,
                gpu_lock_path=gpu_lock_path,
                shared_lock_root=shared_lock_root,
                expected_run_ids=expected_run_ids,
            )
            while True:
                outcome = controller.run_next()
                print(json.dumps({
                    'stage_id': outcome.stage_id,
                    'exit_code': outcome.exit_code,
                    'fingerprint': outcome.fingerprint,
                    'message': outcome.message,
                    'retry_not_before': outcome.retry_not_before,
                    'retry_remaining_seconds': outcome.retry_remaining_seconds,
                }, sort_keys=True))
                if outcome.exit_code != 0:
                    return outcome.exit_code
                if outcome.stage == 'complete':
                    break
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return PERMANENT_EXIT
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
