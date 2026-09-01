#!/usr/bin/env python3
"""Advance the durable optimization campaign under the common GPU lease."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Sequence

sys.dont_write_bytecode = True


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
from mambapose_opt.numeric_conversion import numeric_stage_plan
from mambapose_opt.pwl_selection import CANONICAL_PWL_CANDIDATES


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
            *, device_index: int = 0, heartbeat_interval: float = 30.0,
            pwl_candidates: Sequence[CandidateSpec] = ()):
        self.campaign_root = campaign_root
        self.manifest_path = manifest_path
        self.device_index = device_index
        self.heartbeat_interval = heartbeat_interval
        self.pwl_candidates = tuple(pwl_candidates)

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
        environment.update({
            'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'CUDA_VISIBLE_DEVICES': str(self.device_index),
            'MAMBAPOSE_PHYSICAL_DEVICE_INDEX': str(self.device_index),
        })
        return environment

    @staticmethod
    def _open_attempt_log(path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        try:
            descriptor = os.open(path, flags, 0o640)
        except OSError as error:
            raise FileExistsError(
                f'attempt log must be a nonsymlink regular file: {path}') \
                from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise FileExistsError(
                    f'attempt log must be a regular file: {path}')
            return os.fdopen(descriptor, 'a', encoding='utf-8')
        except Exception:
            os.close(descriptor)
            raise

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
        if stage == 'pwl-selection':
            command = [
                python,
                str(REPO_ROOT / 'tools/optimization/select_pwl_candidate.py'),
                '--manifest', str(self.manifest_path),
                '--output', self._relative(artifact),
            ]
            for item in self.pwl_candidates:
                calibration = (
                    self.campaign_root / item.route / item.id /
                    str(item.seed) / 'calibrate/calibrate.json')
                command.extend([
                    '--calibration',
                    f'{item.id}={self._relative(calibration)}'])
            return command
        if stage == 'smoke-stage-a':
            return [
                python, str(REPO_ROOT / 'tools/optimization/smoke_pwl.py'),
                '--candidate', candidate.id,
                '--manifest', str(self.manifest_path),
                '--output-root', self._relative(artifact.parent),
                '--device-index', str(self.device_index),
            ]
        if stage in {'convert', 'export'}:
            command = [
                python, str(REPO_ROOT / 'tools/optimization/convert_numeric.py'),
                candidate.id, '--stage', stage,
                '--manifest', str(self.manifest_path),
                '--output', self._relative(artifact),
            ]
            if candidate.features.get('numeric_kind') in {
                    'w8a8', 'pwl', 'pwl-combined'}:
                calibration = artifact.parent.parent / 'calibrate/calibrate.json'
                command.extend([
                    '--calibration-artifact', self._relative(calibration)])
            if candidate.features.get('numeric_kind') in {
                    'pwl', 'pwl-combined'}:
                if candidate.features.get('numeric_kind') == 'pwl-combined':
                    selection = REPO_ROOT / candidate.features[
                        'combined_parent_authority']
                else:
                    selection = (
                        self.campaign_root / candidate.route /
                        'pwl-selection/selection.json')
                command.extend([
                    '--selection-artifact', self._relative(selection)])
            return command
        if stage == 'calibrate' and candidate.route == 'ssm-quant-pwl':
            return [
                python, str(REPO_ROOT / 'tools/optimization/calibrate_numeric.py'),
                '--candidate', candidate.id,
                '--manifest', str(self.manifest_path),
                '--policy', str(REPO_ROOT / candidate.config),
                '--output', self._relative(artifact),
            ]
        if stage == 'profile':
            return [
                python, str(REPO_ROOT / 'tools/optimization/profile_model.py'),
                candidate.id, '--manifest', str(self.manifest_path),
                '--output', self._relative(artifact),
            ]
        if (stage == 'compare'
                and candidate.features.get('numeric_kind') == 'pwl-combined'):
            return [
                python,
                str(REPO_ROOT /
                    'tools/optimization/compare_combined_candidate.py'),
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
        if stage in {'train', 'evaluate', 'latency'}:
            common = [
                candidate.id, '--manifest', str(self.manifest_path),
                '--output', self._relative(artifact),
            ]
        return [python, str(REPO_ROOT / 'tools/optimization' / tool), *common]

    def __call__(
            self, candidate: CandidateSpec, stage: str,
            stage_dir: Path, attempt: int) -> StageOutcome:
        artifact = stage_dir / f'{stage}.json'
        if stage == 'pwl-selection':
            artifact = (
                self.campaign_root / candidate.route /
                'pwl-selection/selection.json')
        elif stage == 'smoke-stage-a':
            artifact = stage_dir / 'smoke.json'
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
        with self._open_attempt_log(log_path) as log:
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
        identifiers: Sequence[str], *,
        admit_conditional: bool = False) -> tuple[CandidateSpec, ...]:
    if not identifiers:
        return tuple(
            candidate for candidate in candidates
            if candidate.features.get('auto_run', True) is not False)
    wanted = set(identifiers)
    selected = tuple(item for item in candidates if item.id in wanted)
    missing = wanted - {item.id for item in selected}
    if missing:
        raise CandidateManifestError(
            f'candidate not found: {sorted(missing)}')
    conditional = tuple(
        item.id for item in selected
        if item.features.get('conditional') is True)
    if conditional and not admit_conditional:
        raise CandidateManifestError(
            f'conditional candidates require --admit-conditional: '
            f'{list(conditional)}')
    return selected


def _stages_for_candidate(candidate: CandidateSpec) -> tuple[str, ...]:
    if candidate.route != 'ssm-quant-pwl':
        return ('profile', 'calibrate', 'train', 'evaluate', 'latency',
                'compare')
    kind = candidate.features.get('numeric_kind')
    if not isinstance(kind, str):
        raise CandidateManifestError(
            f'numeric candidate {candidate.id} has no numeric_kind')
    return numeric_stage_plan(
        kind, conditional=candidate.features.get('conditional') is True,
        recovery=candidate.features.get('recovery_candidate') is True)


def _canonical_pwl_candidates(
        candidates: Sequence[CandidateSpec]) -> tuple[CandidateSpec, ...]:
    selected = tuple(
        candidate for candidate in candidates
        if candidate.features.get('numeric_kind') == 'pwl')
    actual = tuple(
        (candidate.id, candidate.features.get('pwl_function'))
        for candidate in selected)
    # Test fixtures may omit the redundant feature, while production manifests
    # must preserve it. Candidate IDs still make the scheduling authority exact.
    actual_ids = tuple(candidate.id for candidate in selected)
    expected_ids = tuple(item[0] for item in CANONICAL_PWL_CANDIDATES)
    if actual_ids != expected_ids:
        raise CandidateManifestError(
            'PWL campaign requires exactly four canonical candidates in order')
    if any(function is not None and function != expected_function
           for (_, function), (_, expected_function) in zip(
               actual, CANONICAL_PWL_CANDIDATES)):
        raise CandidateManifestError(
            'PWL candidate function differs from canonical manifest')
    return selected


_PWL_DOWNSTREAM_STAGES = (
    'convert', 'smoke-stage-a', 'profile', 'evaluate', 'latency')


def _pwl_campaign_phases(
        candidates: Sequence[CandidateSpec], *,
        selected_candidate_id: str | None = None,
        ) -> tuple[tuple[CandidateSpec, tuple[str, ...]], ...]:
    canonical = _canonical_pwl_candidates(candidates)
    phases = tuple((candidate, ('calibrate',)) for candidate in canonical) + (
        (canonical[0], ('pwl-selection',)),)
    if selected_candidate_id is None:
        return phases
    selected = tuple(
        candidate for candidate in canonical
        if candidate.id == selected_candidate_id)
    if len(selected) != 1:
        raise CandidateManifestError(
            'PWL selection chose a non-canonical candidate')
    return phases + ((selected[0], _PWL_DOWNSTREAM_STAGES),)


def _pwl_expected_run_ids(
        candidates: Sequence[CandidateSpec]) -> tuple[str, ...]:
    canonical = _canonical_pwl_candidates(candidates)
    return (
        tuple(f'{candidate.id}:calibrate' for candidate in canonical)
        + (f'{canonical[0].id}:pwl-selection',)
        + tuple(
            f'{candidate.id}:{stage}'
            for candidate in canonical
            for stage in _PWL_DOWNSTREAM_STAGES))


def _advance_controller(controller: OptimizationController) -> int:
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
            return 0


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
    parser.add_argument('--admit-conditional', action='store_true')
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
            load_candidate_manifest(args.manifest), args.candidate,
            admit_conditional=args.admit_conditional)
    except CandidateManifestError as error:
        print(str(error), file=sys.stderr)
        return PERMANENT_EXIT

    try:
        shared_lock_root, gpu_lock_path = _validated_gpu_lock(
            args.gpu_lock_path)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f'cannot derive canonical GPU lock: {error}', file=sys.stderr)
        return PERMANENT_EXIT
    try:
        pwl_candidates = tuple(
            item for item in candidates
            if item.features.get('numeric_kind') == 'pwl')
        regular_candidates = tuple(
            item for item in candidates
            if item.features.get('numeric_kind') != 'pwl')
        expected_run_ids = tuple(
            f'{candidate.id}:{stage}'
            for candidate in regular_candidates
            for stage in _stages_for_candidate(candidate))
        if pwl_candidates:
            expected_run_ids += _pwl_expected_run_ids(pwl_candidates)
        runner = SubprocessStageRunner(
            args.campaign_root, args.manifest, device_index=args.device_index,
            pwl_candidates=pwl_candidates)

        def advance(candidate: CandidateSpec, stages: tuple[str, ...]) -> int:
            controller = OptimizationController(
                args.campaign_root,
                candidate,
                runner,
                repository_root=REPO_ROOT,
                manifest_path=args.manifest,
                device_index=args.device_index,
                gpu_lock_path=gpu_lock_path,
                shared_lock_root=shared_lock_root,
                stages=stages,
                expected_run_ids=expected_run_ids,
            )
            return _advance_controller(controller)

        for candidate in regular_candidates:
            exit_code = advance(candidate, _stages_for_candidate(candidate))
            if exit_code:
                return exit_code
        if pwl_candidates:
            phases = _pwl_campaign_phases(pwl_candidates)
            for candidate, stages in phases:
                exit_code = advance(candidate, stages)
                if exit_code:
                    return exit_code
            selection_path = (
                args.campaign_root / 'ssm-quant-pwl/'
                'pwl-selection/selection.json')
            from mambapose_opt.pwl_selection import (
                validate_pwl_selection_artifact)
            selection = validate_pwl_selection_artifact(
                json.loads(selection_path.read_text(encoding='utf-8')),
                repository_root=REPO_ROOT, manifest_path=args.manifest)
            selected_id = selection.get('selected_candidate_id')
            if selected_id is not None:
                candidate, stages = _pwl_campaign_phases(
                    pwl_candidates,
                    selected_candidate_id=str(selected_id))[-1]
                exit_code = advance(candidate, stages)
                if exit_code:
                    return exit_code
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return PERMANENT_EXIT
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
