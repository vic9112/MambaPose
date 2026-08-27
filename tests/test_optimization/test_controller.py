from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(repo_root):
    from mambapose_opt.schema import CandidateSpec

    config = repo_root / 'configs/candidate.py'
    checkpoint = repo_root / 'checkpoints/parent.pth'
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text('model = dict(type="Fixture")\n')
    checkpoint.write_bytes(b'parent checkpoint')
    return CandidateSpec.from_dict({
        'id': 'fixture',
        'route': 'accuracy-first',
        'kind': 'float',
        'config': 'configs/candidate.py',
        'checkpoint': 'checkpoints/parent.pth',
        'checkpoint_sha256': _sha256(checkpoint),
        'seed': 0,
        'features': {},
    })


def _outcome(stage, artifact, *, exit_code=0, valid=True, fingerprint='ok'):
    from mambapose_opt.controller import StageOutcome

    hashes = {str(artifact): _sha256(artifact)} if artifact.exists() else {}
    return StageOutcome(
        stage_id=f'fixture:{stage}',
        stage=stage,
        candidate_id='fixture',
        exit_code=exit_code,
        fingerprint=fingerprint,
        artifacts_valid=valid,
        artifacts=(artifact,),
        artifact_sha256=hashes,
    )


def _mock_lease(monkeypatch, entered):
    from mambapose_opt import controller
    from mambapose_opt.gpu_guard import GpuLease

    @contextmanager
    def lease(lock_path, device_index, allowed_pids, *, stage_id):
        entered.append((Path(lock_path), device_index, stage_id, set(allowed_pids)))
        yield GpuLease(
            stage_id, os.getpid(), 'boot', 'now', device_index,
            tuple(sorted(allowed_pids)))

    monkeypatch.setattr(controller, 'exclusive_cuda_stage', lease)


def test_stage_completes_only_after_artifacts_validate(tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / 'metrics.json'
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"AP": 72.8}\n')
        return _outcome(stage, artifact, valid=False)

    campaign = tmp_path / 'work_dirs/optimization'
    outcome = OptimizationController(
        campaign, candidate, runner, repository_root=tmp_path,
        stages=('evaluate',)).run_next()

    assert outcome.exit_code == 78
    run = StateStore(campaign).read()['runs']['fixture:evaluate']
    assert run['status'] == 'blocked'
    assert entered[0][2] == 'fixture:evaluate'


def test_contention_returns_75_and_retry_lineage_is_append_only(
        tmp_path, monkeypatch):
    from mambapose_opt import controller
    from mambapose_opt.controller import OptimizationController
    from mambapose_opt.gpu_guard import ExternalGpuContention
    from mambapose_repro.state import StateStore

    candidate = _candidate(tmp_path)

    @contextmanager
    def contended(*args, **kwargs):
        raise ExternalGpuContention('external pid=9001')
        yield

    monkeypatch.setattr(controller, 'exclusive_cuda_stage', contended)
    campaign = tmp_path / 'work_dirs/optimization'
    instance = OptimizationController(
        campaign, candidate, lambda *args: None,
        repository_root=tmp_path, stages=('train',))

    first = instance.run_next()
    second = instance.run_next()

    assert first.exit_code == second.exit_code == 75
    run = StateStore(campaign).read()['runs']['fixture:train']
    assert run['status'] == 'retry_wait'
    assert [item['attempt'] for item in run['retry_lineage']] == [1, 2]
    assert all(item['exit_code'] == 75 for item in run['retry_lineage'])
    events = [
        json.loads(line)
        for line in (campaign / 'events.jsonl').read_text().splitlines()
    ]
    waits = [event for event in events if event['status'] == 'retry_wait']
    assert [event['attempt'] for event in waits] == [1, 2]


def test_invalid_config_or_checkpoint_hash_returns_78_without_runner(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController
    from mambapose_repro.orchestrator import failure_fingerprint

    candidate = _candidate(tmp_path)
    (tmp_path / candidate.checkpoint).write_bytes(b'tampered')
    entered = []
    _mock_lease(monkeypatch, entered)
    calls = []

    result = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate,
        lambda *args: calls.append(args), repository_root=tmp_path,
        stages=('profile',)).run_next()

    assert result.exit_code == 78
    assert 'checkpoint' in result.message
    assert result.fingerprint == failure_fingerprint(result.message)
    assert calls == []
    assert entered == []


def test_all_cuda_stages_share_one_lease_and_compare_is_cpu_only(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        artifact = stage_dir / f'{stage}.json'
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{}\n')
        return _outcome(stage, artifact)

    instance = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('profile', 'compare'))

    assert instance.run_next().exit_code == 0
    assert instance.run_next().exit_code == 0
    assert [item[2] for item in entered] == ['fixture:profile']
    assert entered[0][0].name == 'gpu.lock'


def test_checkpoint_retention_keeps_best_and_two_valid_resume_points(tmp_path):
    from mambapose_opt.controller import retain_checkpoints

    paths = []
    for name in ('best_AP_epoch_4.pth', 'epoch_1.pth', 'epoch_2.pth',
                 'epoch_3.pth', 'epoch_4.pth'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)

    kept = retain_checkpoints(
        tmp_path,
        validator=lambda path: path.name not in {'epoch_3.pth'},
    )

    assert {path.name for path in kept} == {
        'best_AP_epoch_4.pth', 'epoch_4.pth', 'epoch_2.pth'}
    assert {path.name for path in tmp_path.glob('*.pth')} == {
        'best_AP_epoch_4.pth', 'epoch_4.pth', 'epoch_2.pth'}


def test_completed_train_stage_applies_checkpoint_retention(
        tmp_path, monkeypatch):
    from mambapose_opt.controller import OptimizationController

    candidate = _candidate(tmp_path)
    entered = []
    _mock_lease(monkeypatch, entered)

    def runner(candidate, stage, stage_dir, attempt):
        for name in (
                'best_AP_epoch_4.pth', 'epoch_1.pth', 'epoch_2.pth',
                'epoch_3.pth', 'epoch_4.pth'):
            (stage_dir / name).write_bytes(name.encode())
        artifact = stage_dir / 'train.json'
        artifact.write_text('{}\n')
        return _outcome(stage, artifact)

    instance = OptimizationController(
        tmp_path / 'work_dirs/optimization', candidate, runner,
        repository_root=tmp_path, stages=('train',),
        checkpoint_validator=lambda path: path.name != 'epoch_3.pth')

    result = instance.run_next()

    checkpoint_dir = (
        tmp_path / 'work_dirs/optimization/accuracy-first/fixture/0/train')
    assert result.exit_code == 0
    assert {path.name for path in checkpoint_dir.glob('*.pth')} == {
        'best_AP_epoch_4.pth', 'epoch_4.pth', 'epoch_2.pth'}
