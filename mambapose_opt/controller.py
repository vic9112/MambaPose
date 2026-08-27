"""Durable, one-stage-at-a-time optimization campaign control."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from mambapose_repro.orchestrator import (
    PERMANENT_EXIT,
    TRANSIENT_EXIT,
    failure_fingerprint,
)
from mambapose_repro.checkpoint import validate_checkpoint
from mambapose_repro.state import StateStore

from .gpu_guard import (
    ConcurrentCudaStage,
    ExternalGpuContention,
    GpuLease,
    exclusive_cuda_stage,
)
from .schema import CandidateSpec


STAGES = ('profile', 'calibrate', 'train', 'evaluate', 'latency', 'compare')
CUDA_STAGES = frozenset(STAGES) - {'compare'}
StageRunner = Callable[[CandidateSpec, str, Path, int], 'StageOutcome']
ArtifactValidator = Callable[[str, 'StageOutcome'], bool]


@dataclass(frozen=True)
class StageOutcome:
    stage_id: str
    stage: str
    candidate_id: str
    exit_code: int
    fingerprint: str
    artifacts_valid: bool = False
    artifacts: tuple[Path, ...] = ()
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    attempt: int = 0
    message: str = ''
    gpu_lease: GpuLease | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_order(path: Path) -> tuple[int, int, str]:
    match = re.search(r'(?:epoch|iter)_(\d+)', path.name)
    sequence = int(match.group(1)) if match else -1
    try:
        modified = path.stat().st_mtime_ns
    except OSError:
        modified = -1
    return sequence, modified, path.name


def retain_checkpoints(
        checkpoint_dir: Path,
        *,
        validator: Callable[[Path], bool],
        resume_limit: int = 2) -> tuple[Path, ...]:
    """Retain one valid best checkpoint and two valid resume points."""
    checkpoint_dir = Path(checkpoint_dir)
    candidates = tuple(checkpoint_dir.glob('*.pth'))
    valid = tuple(path for path in candidates if validator(path))
    best = sorted(
        (path for path in valid if path.name.startswith('best_')),
        key=_checkpoint_order,
        reverse=True,
    )[:1]
    resumes = sorted(
        (path for path in valid if path.name.startswith(('epoch_', 'iter_'))),
        key=_checkpoint_order,
        reverse=True,
    )[:resume_limit]
    kept = tuple(best + resumes)
    keep_set = set(kept)
    for path in candidates:
        if path not in keep_set:
            path.unlink()
    return kept


class OptimizationController:
    """Single writer that advances exactly one validated stage per call."""

    def __init__(
            self,
            root: Path,
            candidate: CandidateSpec,
            runner: StageRunner,
            *,
            repository_root: Path | None = None,
            stages: Sequence[str] = STAGES,
            device_index: int = 0,
            allowed_pids: Sequence[int] | None = None,
            artifact_validator: ArtifactValidator | None = None,
            checkpoint_validator: Callable[[Path], bool] | None = None):
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f'unknown optimization stages: {sorted(unknown)}')
        if len(stages) != len(set(stages)):
            raise ValueError('optimization stages must be unique')
        self.root = Path(root)
        self.candidate = candidate
        self.runner = runner
        self.repository_root = Path(repository_root or Path.cwd()).resolve()
        self.stages = tuple(stages)
        self.device_index = device_index
        self.allowed_pids = tuple(allowed_pids or (os.getpid(),))
        self.artifact_validator = artifact_validator
        self.checkpoint_validator = (
            checkpoint_validator or self._checkpoint_is_valid)
        self.store = StateStore(self.root)

    @staticmethod
    def _checkpoint_is_valid(path: Path) -> bool:
        requires_training_state = path.name.startswith(('epoch_', 'iter_'))
        return validate_checkpoint(
            path, require_training_state=requires_training_state).valid

    def _stage_id(self, stage: str) -> str:
        return f'{self.candidate.id}:{stage}'

    def _stage_dir(self, stage: str) -> Path:
        return (
            self.root / self.candidate.route / self.candidate.id /
            str(self.candidate.seed) / stage)

    def _next_stage(self) -> str | None:
        runs = self.store.read().get('runs', {})
        for stage in self.stages:
            if runs.get(self._stage_id(stage), {}).get('status') != 'complete':
                return stage
        return None

    def _preflight_error(self) -> str | None:
        paths = {
            'config': self.candidate.config,
            'checkpoint': self.candidate.checkpoint,
        }
        resolved: dict[str, Path] = {}
        for label, relative in paths.items():
            path = (self.repository_root / relative).resolve()
            try:
                path.relative_to(self.repository_root)
            except ValueError:
                return f'{label} escapes repository root for {self.candidate.id}'
            if not path.is_file():
                return f'{label} is missing for {self.candidate.id}: {relative}'
            resolved[label] = path
        try:
            actual = _sha256(resolved['checkpoint'])
        except OSError as error:
            return f'cannot hash checkpoint for {self.candidate.id}: {error}'
        if actual != self.candidate.checkpoint_sha256:
            return f'checkpoint sha256 mismatch for {self.candidate.id}'
        return None

    def _failure(
            self, stage: str, attempt: int, exit_code: int,
            message: str) -> StageOutcome:
        return StageOutcome(
            stage_id=self._stage_id(stage),
            stage=stage,
            candidate_id=self.candidate.id,
            exit_code=exit_code,
            fingerprint=failure_fingerprint(message),
            attempt=attempt,
            message=message,
        )

    def _artifacts_are_valid(self, stage: str, outcome: StageOutcome) -> bool:
        if not outcome.artifacts_valid or not outcome.artifacts:
            return False
        for supplied in outcome.artifacts:
            path = Path(supplied).resolve()
            try:
                path.relative_to(self.root.resolve())
            except ValueError:
                return False
            if not path.is_file():
                return False
            expected = outcome.artifact_sha256.get(str(supplied))
            if expected is None or expected != _sha256(path):
                return False
        return (
            self.artifact_validator(stage, outcome)
            if self.artifact_validator is not None else True)

    def _record(self, outcome: StageOutcome, status: str) -> None:
        previous = self.store.read().get('runs', {}).get(outcome.stage_id, {})
        lineage = list(previous.get('retry_lineage', []))
        lineage.append({
            'attempt': outcome.attempt,
            'exit_code': outcome.exit_code,
            'fingerprint': outcome.fingerprint,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        details = {
            'retry_lineage': lineage,
            'failure_fingerprint': outcome.fingerprint,
            'exit_code': outcome.exit_code,
        }
        if outcome.message:
            details['message'] = outcome.message
        if outcome.gpu_lease is not None:
            details['gpu_lease'] = {
                'stage_id': outcome.gpu_lease.stage_id,
                'pid': outcome.gpu_lease.pid,
                'boot_id': outcome.gpu_lease.boot_id,
                'timestamp': outcome.gpu_lease.timestamp,
                'device_index': outcome.gpu_lease.device_index,
                'allowed_pids': list(outcome.gpu_lease.allowed_pids),
            }
        self.store.transition(
            outcome.stage_id, status, attempt=outcome.attempt, **details)

    def run_next(self) -> StageOutcome:
        """Run the next incomplete stage and persist its terminal disposition."""
        stage = self._next_stage()
        if stage is None:
            return StageOutcome(
                stage_id=f'{self.candidate.id}:complete',
                stage='complete',
                candidate_id=self.candidate.id,
                exit_code=0,
                fingerprint='validated-campaign-completion',
                artifacts_valid=True,
            )
        stage_id = self._stage_id(stage)
        previous = self.store.read().get('runs', {}).get(stage_id, {})
        attempt = int(previous.get('attempt', 0)) + 1
        self.store.transition(stage_id, 'running', attempt=attempt)

        error = self._preflight_error()
        if error is not None:
            outcome = self._failure(stage, attempt, PERMANENT_EXIT, error)
            self._record(outcome, 'blocked')
            return outcome

        stage_dir = self._stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        lease: GpuLease | None = None
        try:
            if stage in CUDA_STAGES:
                with exclusive_cuda_stage(
                        self.root / 'gpu.lock', self.device_index,
                        self.allowed_pids, stage_id=stage_id) as acquired:
                    lease = acquired
                    outcome = self.runner(
                        self.candidate, stage, stage_dir, attempt)
            else:
                outcome = self.runner(
                    self.candidate, stage, stage_dir, attempt)
        except (ExternalGpuContention, ConcurrentCudaStage) as contention:
            outcome = self._failure(
                stage, attempt, TRANSIENT_EXIT, str(contention))
            self._record(outcome, 'retry_wait')
            return outcome
        except Exception as error:  # fail closed around route-owned runners
            message = f'{type(error).__name__}: {error}'
            outcome = self._failure(
                stage, attempt, PERMANENT_EXIT, message)
            self._record(outcome, 'blocked')
            return outcome

        if not isinstance(outcome, StageOutcome):
            outcome = self._failure(
                stage, attempt, PERMANENT_EXIT,
                f'runner returned invalid outcome: {type(outcome).__name__}')
        else:
            outcome = replace(outcome, attempt=attempt, gpu_lease=lease)
        if (
                outcome.stage != stage
                or outcome.stage_id != stage_id
                or outcome.candidate_id != self.candidate.id):
            outcome = self._failure(
                stage, attempt, PERMANENT_EXIT,
                'runner outcome identity does not match requested stage')

        if outcome.exit_code == TRANSIENT_EXIT:
            self._record(outcome, 'retry_wait')
            return outcome
        if outcome.exit_code != 0:
            normalized = (
                outcome if outcome.exit_code == PERMANENT_EXIT else
                replace(outcome, exit_code=PERMANENT_EXIT))
            self._record(normalized, 'blocked')
            return normalized
        if not self._artifacts_are_valid(stage, outcome):
            invalid = self._failure(
                stage, attempt, PERMANENT_EXIT,
                f'{stage_id} artifact validation failed')
            self._record(invalid, 'blocked')
            return invalid

        if stage == 'train':
            checkpoint_dir = stage_dir / 'checkpoints'
            if not checkpoint_dir.is_dir():
                checkpoint_dir = stage_dir
            try:
                retain_checkpoints(
                    checkpoint_dir, validator=self.checkpoint_validator)
            except OSError as error:
                invalid = self._failure(
                    stage, attempt, PERMANENT_EXIT,
                    f'{stage_id} checkpoint retention failed: {error}')
                self._record(invalid, 'blocked')
                return invalid

        self._record(outcome, 'complete')
        return outcome
