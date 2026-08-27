"""Durable, one-stage-at-a-time optimization campaign control."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from mambapose_repro.orchestrator import (
    CampaignLock,
    ConcurrentCampaign,
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
    artifact_evidence: tuple[Mapping[str, str], ...] = ()
    retry_not_before: str | None = None
    retry_remaining_seconds: float | None = None


class ArtifactValidationError(ValueError):
    """Raised when a stage artifact cannot prove its own identity."""


class CheckpointRetentionError(ValueError):
    """Raised before mutation when the bounded resume set is not provable."""


class CampaignPlanError(ValueError):
    """Raised when the immutable expected-run plan changes or is malformed."""


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


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonempty_integer_map(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(key, str) and bool(key)
            and _nonnegative_integer(item)
            for key, item in value.items())
    )


def _output_shape_tree(value: object) -> bool:
    if isinstance(value, list):
        if not value:
            return False
        if all(_nonnegative_integer(item) for item in value):
            return True
        if any(isinstance(item, (int, bool)) for item in value):
            return False
        return all(_output_shape_tree(item) for item in value)
    if isinstance(value, dict):
        return (
            bool(value)
            and all(isinstance(key, str) for key in value)
            and all(_output_shape_tree(item) for item in value.values())
        )
    return isinstance(value, str) and bool(value)


def retain_checkpoints(
        checkpoint_dir: Path,
        *,
        validator: Callable[[Path], bool],
        resume_limit: int = 2) -> tuple[Path, ...]:
    """Retain one valid best checkpoint and two valid resume points."""
    checkpoint_dir = Path(checkpoint_dir)
    best_candidates = tuple(checkpoint_dir.glob('best_*.pth'))
    resume_candidates = tuple(
        path for pattern in ('epoch_*.pth', 'iter_*.pth')
        for path in checkpoint_dir.glob(pattern))
    recognized = best_candidates + resume_candidates
    try:
        valid = tuple(path for path in recognized if validator(path))
    except Exception as error:
        raise CheckpointRetentionError(
            f'checkpoint validation failed: {error}') from error
    best = sorted(
        (path for path in valid if path in best_candidates),
        key=_checkpoint_order,
        reverse=True,
    )[:1]
    resumes = sorted(
        (path for path in valid if path in resume_candidates),
        key=_checkpoint_order,
        reverse=True,
    )[:resume_limit]
    if len(best) != 1:
        raise CheckpointRetentionError(
            'training completion requires one valid best checkpoint')
    if len(resumes) != resume_limit:
        raise CheckpointRetentionError(
            f'training completion requires two valid resume checkpoints; '
            f'found {len(resumes)}')
    kept = tuple(best + resumes)
    keep_set = set(kept)
    for path in recognized:
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
            gpu_lock_path: Path | None = None,
            shared_lock_root: Path | None = None,
            controller_lock_path: Path | None = None,
            allowed_pids: Sequence[int] | None = None,
            artifact_validator: ArtifactValidator | None = None,
            checkpoint_validator: Callable[[Path], bool] | None = None,
            expected_run_ids: Sequence[str] | None = None,
            max_attempts: int = 3,
            retry_delays: Sequence[int] = (30, 120, 600),
            now: Callable[[], datetime] | None = None):
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f'unknown optimization stages: {sorted(unknown)}')
        if len(stages) != len(set(stages)):
            raise ValueError('optimization stages must be unique')
        if max_attempts <= 0:
            raise ValueError('max_attempts must be positive')
        if not retry_delays or any(delay < 0 for delay in retry_delays):
            raise ValueError('retry_delays must contain non-negative values')
        self.candidate = candidate
        self.runner = runner
        self.repository_root = Path(repository_root or Path.cwd()).resolve()
        self.root = Path(root).resolve()
        try:
            self.root.relative_to(self.repository_root)
        except ValueError as error:
            raise ValueError(
                'optimization campaign root must stay inside the repository') from error
        if self.root.name != 'optimization' or self.root.parent.name != 'work_dirs':
            raise ValueError(
                'optimization campaign root must end in work_dirs/optimization')
        self.stages = tuple(stages)
        self.device_index = device_index
        self.shared_lock_root = Path(
            shared_lock_root or self.repository_root).resolve()
        canonical_gpu_lock = (
            self.shared_lock_root / 'work_dirs/optimization/gpu.lock').resolve()
        self.gpu_lock_path = Path(
            gpu_lock_path or canonical_gpu_lock).resolve()
        if self.gpu_lock_path != canonical_gpu_lock:
            raise ValueError(
                'optimization GPU lock must equal the trusted canonical path')
        self.controller_lock_path = Path(
            controller_lock_path or (self.root / 'controller.lock')).resolve()
        if self.controller_lock_path != self.root / 'controller.lock':
            raise ValueError(
                'optimization controller lock must stay at '
                'campaign-root/controller.lock')
        self.allowed_pids = tuple(allowed_pids or (os.getpid(),))
        self.artifact_validator = artifact_validator
        self.checkpoint_validator = (
            checkpoint_validator or self._checkpoint_is_valid)
        self.expected_run_ids = tuple(
            expected_run_ids or (
                f'{self.candidate.id}:{stage}' for stage in self.stages))
        self.max_attempts = max_attempts
        self.retry_delays = tuple(retry_delays)
        self.now = now or (lambda: datetime.now(timezone.utc))
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
            run = runs.get(self._stage_id(stage), {})
            if (
                    run.get('status') != 'complete'
                    or not self._completed_evidence_valid(stage, run)):
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

    def _artifact_schema(self, stage: str, path: Path) -> str:
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactValidationError(
                f'{stage} artifact is not valid JSON: {path}: {error}') from error
        if not isinstance(value, dict):
            raise ArtifactValidationError(
                f'{stage} artifact root must be an object: {path}')
        if stage == 'profile':
            required = {
                'schema_version', 'git_commit', 'candidate', 'config',
                'checkpoint', 'checkpoint_sha256', 'input_shapes',
                'output_shapes', 'parameters', 'modules',
            }
            if set(value) != required:
                raise ArtifactValidationError(
                    'profile artifact fields do not match profile schema v1')
            if value.get('schema_version') != 1:
                raise ArtifactValidationError(
                    'profile artifact schema_version must be 1')
            if not (
                    isinstance(value.get('git_commit'), str)
                    and re.fullmatch(r'[0-9a-f]{40}', value['git_commit'])):
                raise ArtifactValidationError(
                    'profile artifact git_commit must be a full commit hash')
            if value.get('candidate') != self.candidate.id:
                raise ArtifactValidationError(
                    'profile artifact candidate identity mismatch')
            if value.get('config') != self.candidate.config.as_posix():
                raise ArtifactValidationError(
                    'profile artifact config identity mismatch')
            if value.get('checkpoint') != self.candidate.checkpoint.as_posix():
                raise ArtifactValidationError(
                    'profile artifact checkpoint identity mismatch')
            if value.get('checkpoint_sha256') != self.candidate.checkpoint_sha256:
                raise ArtifactValidationError(
                    'profile artifact checkpoint hash mismatch')
            input_shape = value.get('input_shapes')
            if not (
                    isinstance(input_shape, list)
                    and len(input_shape) == 4
                    and all(
                        isinstance(dimension, int)
                        and not isinstance(dimension, bool)
                        and dimension > 0
                        for dimension in input_shape)):
                raise ArtifactValidationError(
                    'profile artifact input_shapes must be four positive integers')
            if not _output_shape_tree(value.get('output_shapes')):
                raise ArtifactValidationError(
                    'profile artifact output_shapes tree is invalid')
            parameters = value.get('parameters')
            parameter_fields = {
                'total', 'trainable', 'bytes_by_dtype', 'by_prefix'}
            if not (
                    isinstance(parameters, dict)
                    and set(parameters) == parameter_fields
                    and _nonnegative_integer(parameters.get('total'))
                    and _nonnegative_integer(parameters.get('trainable'))
                    and parameters['trainable'] <= parameters['total']
                    and _nonempty_integer_map(parameters.get('bytes_by_dtype'))
                    and _nonempty_integer_map(parameters.get('by_prefix'))):
                raise ArtifactValidationError(
                    'profile artifact parameters object is invalid')
            modules = value.get('modules')
            if not (
                    isinstance(modules, list)
                    and bool(modules)
                    and all(
                        isinstance(record, dict)
                        and set(record) == {
                            'name', 'kind', 'parameters', 'hazard'}
                        and isinstance(record.get('name'), str)
                        and isinstance(record.get('kind'), str)
                        and bool(record['kind'])
                        and _nonnegative_integer(record.get('parameters'))
                        and (
                            record.get('hazard') is None
                            or isinstance(record.get('hazard'), str))
                        for record in modules)):
                raise ArtifactValidationError(
                    'profile artifact modules list is invalid')
            return 'optimization-profile-v1'

        required = {'schema_version', 'candidate_id', 'stage', 'result'}
        if set(value) != required:
            raise ArtifactValidationError(
                f'{stage} artifact must use the versioned stage envelope')
        if value.get('schema_version') != 1:
            raise ArtifactValidationError(
                f'{stage} artifact schema_version must be 1')
        if value.get('candidate_id') != self.candidate.id:
            raise ArtifactValidationError(
                f'{stage} artifact candidate identity mismatch')
        if value.get('stage') != stage:
            raise ArtifactValidationError(
                f'{stage} artifact stage identity mismatch')
        if not isinstance(value.get('result'), dict):
            raise ArtifactValidationError(
                f'{stage} artifact result must be an object')
        return 'optimization-stage-envelope-v1'

    def _validate_artifacts(
            self, stage: str, outcome: StageOutcome
            ) -> tuple[Mapping[str, str], ...]:
        if not outcome.artifacts_valid or not outcome.artifacts:
            raise ArtifactValidationError(
                f'{stage} runner did not supply validated artifacts')
        evidence: list[Mapping[str, str]] = []
        for supplied in outcome.artifacts:
            path = Path(supplied).resolve()
            try:
                path.relative_to(self.root.resolve())
            except ValueError:
                raise ArtifactValidationError(
                    f'{stage} artifact escapes the campaign root: {path}')
            if not path.is_file():
                raise ArtifactValidationError(
                    f'{stage} artifact is missing: {path}')
            expected = outcome.artifact_sha256.get(str(supplied))
            actual = _sha256(path)
            if expected is None or expected != actual:
                raise ArtifactValidationError(
                    f'{stage} artifact sha256 mismatch: {path}')
            schema = self._artifact_schema(stage, path)
            evidence.append({
                'path': path.relative_to(self.root.resolve()).as_posix(),
                'sha256': actual,
                'schema': schema,
            })
        if (
                self.artifact_validator is not None
                and not self.artifact_validator(stage, outcome)):
            raise ArtifactValidationError(
                f'{stage} additional artifact validation failed')
        return tuple(evidence)

    def _completed_evidence_valid(
            self, stage: str, run: Mapping[str, object]) -> bool:
        evidence = run.get('artifact_evidence')
        if not isinstance(evidence, list) or not evidence:
            return False
        artifacts: list[Path] = []
        hashes: dict[str, str] = {}
        try:
            for record in evidence:
                if not isinstance(record, dict):
                    return False
                relative = Path(record['path'])
                if relative.is_absolute() or any(
                        part in {'.', '..'} for part in relative.parts):
                    return False
                path = (self.root / relative).resolve()
                path.relative_to(self.root.resolve())
                if not path.is_file() or _sha256(path) != record['sha256']:
                    return False
                if self._artifact_schema(stage, path) != record['schema']:
                    return False
                artifacts.append(path)
                hashes[str(path)] = record['sha256']
            outcome = StageOutcome(
                stage_id=self._stage_id(stage),
                stage=stage,
                candidate_id=self.candidate.id,
                exit_code=0,
                fingerprint=str(run.get('completion_fingerprint', 'complete')),
                artifacts_valid=True,
                artifacts=tuple(artifacts),
                artifact_sha256=hashes,
            )
            return (
                self.artifact_validator(stage, outcome)
                if self.artifact_validator is not None else True)
        except Exception:
            return False

    def _record(
            self, outcome: StageOutcome, status: str,
            **status_details: object) -> None:
        previous = self.store.read().get('runs', {}).get(outcome.stage_id, {})
        lineage = list(previous.get('retry_lineage', []))
        lineage_record = {
            'attempt': outcome.attempt,
            'status': status,
            'exit_code': outcome.exit_code,
            'fingerprint': outcome.fingerprint,
            'timestamp': self.now().isoformat(),
            **status_details,
        }
        lineage.append(lineage_record)
        details = {
            'retry_lineage': lineage,
            'failure_fingerprint': outcome.fingerprint,
            'exit_code': outcome.exit_code,
            'artifact_evidence': [dict(item) for item in outcome.artifact_evidence],
            **status_details,
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

    def _start_attempt(self, stage_id: str, attempt: int) -> None:
        previous = self.store.read().get('runs', {}).get(stage_id, {})
        lineage = list(previous.get('retry_lineage', []))
        lineage.append({
            'attempt': attempt,
            'status': 'started',
            'timestamp': self.now().isoformat(),
        })
        self.store.transition(
            stage_id, 'running', attempt=attempt, retry_lineage=lineage)

    def _finish_transient(self, outcome: StageOutcome) -> StageOutcome:
        if outcome.attempt < self.max_attempts:
            delay = self.retry_delays[min(
                outcome.attempt - 1, len(self.retry_delays) - 1)]
            not_before = (self.now() + timedelta(seconds=delay)).isoformat()
            self._record(
                outcome, 'retry_wait', retry_delay_seconds=delay,
                retry_not_before=not_before)
            return replace(
                outcome,
                retry_not_before=not_before,
                retry_remaining_seconds=float(delay),
            )
        exhausted = self._failure(
            outcome.stage,
            outcome.attempt,
            PERMANENT_EXIT,
            f'{outcome.stage_id} exhausted {self.max_attempts} attempts; '
            f'last failure {outcome.fingerprint}',
        )
        self._record(exhausted, 'exhausted')
        return exhausted

    def _pending_retry(
            self, stage: str,
            previous: Mapping[str, object]) -> StageOutcome | None:
        if previous.get('status') != 'retry_wait':
            return None
        value = previous.get('retry_not_before')
        try:
            not_before = datetime.fromisoformat(str(value))
            if not_before.tzinfo is None:
                raise ValueError('deadline must include a timezone')
        except (TypeError, ValueError) as error:
            outcome = self._failure(
                stage, int(previous.get('attempt', 0)), PERMANENT_EXIT,
                f'{self._stage_id(stage)} has invalid retry deadline: {value!r}')
            self._record(outcome, 'blocked')
            return outcome
        remaining = (not_before - self.now()).total_seconds()
        if remaining <= 0:
            return None
        return replace(
            self._failure(
                stage, int(previous.get('attempt', 0)), TRANSIENT_EXIT,
                f'{self._stage_id(stage)} retry waits until '
                f'{not_before.isoformat()}'),
            retry_not_before=not_before.isoformat(),
            retry_remaining_seconds=remaining,
        )

    def _ensure_campaign_plan(self) -> None:
        expected = {
            'schema_version': 1,
            'run_ids': list(self.expected_run_ids),
        }
        path = self.root / 'campaign-plan.json'
        if path.exists():
            try:
                actual = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as error:
                raise CampaignPlanError(
                    f'campaign plan is unreadable: {error}') from error
            if actual != expected:
                raise CampaignPlanError(
                    f'campaign plan mismatch: expected {expected}, got {actual}')
            return
        self.store._atomic_json(path, expected)

    def run_next(self) -> StageOutcome:
        """Run the next incomplete stage and persist its terminal disposition."""
        try:
            with CampaignLock(self.controller_lock_path):
                return self._run_next_locked()
        except ConcurrentCampaign as error:
            stage = self.stages[0]
            return self._failure(stage, 0, TRANSIENT_EXIT, str(error))

    def _run_next_locked(self) -> StageOutcome:
        """Advance state while holding the controller-wide writer lock."""
        try:
            self._ensure_campaign_plan()
        except CampaignPlanError as error:
            stage = self.stages[0]
            stage_id = self._stage_id(stage)
            previous = self.store.read().get('runs', {}).get(stage_id, {})
            attempt = int(previous.get('attempt', 0)) + 1
            self._start_attempt(stage_id, attempt)
            outcome = self._failure(
                stage, attempt, PERMANENT_EXIT, str(error))
            self._record(outcome, 'blocked')
            return outcome
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
        pending_retry = self._pending_retry(stage, previous)
        if pending_retry is not None:
            return pending_retry
        previous_attempt = int(previous.get('attempt', 0))
        if previous_attempt >= self.max_attempts:
            exhausted = self._failure(
                stage,
                previous_attempt,
                PERMANENT_EXIT,
                f'{stage_id} exhausted {self.max_attempts} attempts',
            )
            self._record(exhausted, 'exhausted')
            return exhausted
        attempt = previous_attempt + 1
        self._start_attempt(stage_id, attempt)

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
                        self.gpu_lock_path, self.device_index,
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
            return self._finish_transient(outcome)
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
            return self._finish_transient(outcome)
        if outcome.exit_code != 0:
            normalized = (
                outcome if outcome.exit_code == PERMANENT_EXIT else
                replace(outcome, exit_code=PERMANENT_EXIT))
            self._record(normalized, 'blocked')
            return normalized
        try:
            evidence = self._validate_artifacts(stage, outcome)
        except Exception as error:
            invalid = self._failure(
                stage, attempt, PERMANENT_EXIT,
                f'{stage_id} artifact validation failed: {error}')
            self._record(invalid, 'blocked')
            return invalid
        outcome = replace(outcome, artifact_evidence=evidence)

        if stage == 'train':
            checkpoint_dir = stage_dir / 'checkpoints'
            if not checkpoint_dir.is_dir():
                checkpoint_dir = stage_dir
            try:
                retain_checkpoints(
                    checkpoint_dir, validator=self.checkpoint_validator)
            except Exception as error:
                invalid = self._failure(
                    stage, attempt, PERMANENT_EXIT,
                    f'{stage_id} checkpoint retention failed: {error}')
                self._record(invalid, 'blocked')
                return invalid

        self._record(outcome, 'complete')
        return outcome
