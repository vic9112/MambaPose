from datetime import datetime, timedelta, timezone
import json
import os


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob('*') if path.is_file()
    }


def test_observer_is_strictly_read_only(tmp_path, monkeypatch):
    from mambapose_opt import observe as observer
    from mambapose_opt.observe import observe
    from mambapose_repro.state import StateStore

    store = StateStore(tmp_path)
    store.transition('fixture:train', 'running', attempt=1)
    (tmp_path / 'heartbeat.json').write_text(json.dumps({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(),
        'stage_id': 'fixture:train',
    }))
    monkeypatch.setattr(observer, 'query_compute_processes', lambda _: ())
    before = _snapshot(tmp_path)

    status = observe(tmp_path, heartbeat_max_age=180)

    assert status['health'] == 'running'
    assert _snapshot(tmp_path) == before


def test_observer_reports_stale_heartbeat_without_controlling_processes(
        tmp_path, monkeypatch):
    from mambapose_opt import observe as observer
    from mambapose_opt.observe import observe
    from mambapose_repro.state import StateStore

    StateStore(tmp_path).transition('fixture:train', 'running', attempt=1)
    (tmp_path / 'heartbeat.json').write_text(json.dumps({
        'timestamp': (
            datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        'pid': os.getpid(),
        'stage_id': 'fixture:train',
    }))
    monkeypatch.setattr(observer, 'query_compute_processes', lambda _: ())

    status = observe(tmp_path, heartbeat_max_age=10)

    assert status['health'] == 'stalled'
    assert status['process_alive'] is True


def test_observer_marks_external_gpu_contention(tmp_path, monkeypatch):
    from mambapose_opt import observe as observer
    from mambapose_opt.gpu_guard import GpuProcess
    from mambapose_opt.observe import observe

    (tmp_path / 'gpu.lock').write_text(json.dumps({
        'stage_id': 'fixture:latency',
        'pid': 100,
        'allowed_pids': [100, 101],
        'device_index': 0,
    }))
    monkeypatch.setattr(observer, 'query_compute_processes', lambda _: (
        GpuProcess(101, 20, 'worker'),
        GpuProcess(9001, 4096, 'foreign.py'),
    ))

    status = observe(tmp_path, heartbeat_max_age=180)

    assert status['gpu_contention'] is True
    assert status['external_gpu_pids'] == [9001]


def test_observer_recomputes_active_controller_descendants(
        tmp_path, monkeypatch):
    from mambapose_opt import observe as observer
    from mambapose_opt.gpu_guard import GpuProcess
    from mambapose_opt.observe import observe

    (tmp_path / 'gpu.lock').write_text(json.dumps({
        'stage_id': 'fixture:evaluate',
        'pid': 100,
        'allowed_pids': [100],
        'device_index': 0,
    }))
    monkeypatch.setattr(
        observer, 'controller_process_tree', lambda roots: (100, 101))
    monkeypatch.setattr(observer, 'query_compute_processes', lambda _: (
        GpuProcess(101, 1024, 'spawned-cuda-worker'),
    ))

    status = observe(tmp_path, heartbeat_max_age=180)

    assert status['gpu_contention'] is False
    assert status['external_gpu_pids'] == []


def test_observer_can_read_canonical_lock_outside_campaign_root(
        tmp_path, monkeypatch):
    from mambapose_opt import observe as observer
    from mambapose_opt.gpu_guard import GpuProcess
    from mambapose_opt.observe import observe

    campaign = tmp_path / 'route/work_dirs/optimization'
    campaign.mkdir(parents=True)
    lock_path = tmp_path / 'work_dirs/optimization/gpu.lock'
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({
        'stage_id': 'fixture:latency',
        'pid': 100,
        'allowed_pids': [100],
        'device_index': 1,
    }))
    monkeypatch.setattr(
        observer, 'controller_process_tree', lambda roots: (100, 101))
    monkeypatch.setattr(observer, 'query_compute_processes', lambda device: (
        GpuProcess(101, 256, f'worker-on-{device}'),
    ))

    status = observe(
        campaign, heartbeat_max_age=180, gpu_lock_path=lock_path)

    assert status['gpu_contention'] is False
    assert status['gpu_processes'][0]['command'] == 'worker-on-1'


def test_campaign_complete_requires_every_planned_run(tmp_path, monkeypatch):
    from mambapose_opt import observe as observer
    from mambapose_opt.observe import observe
    from mambapose_repro.state import StateStore

    (tmp_path / 'campaign-plan.json').write_text(json.dumps({
        'schema_version': 1,
        'run_ids': ['first:evaluate', 'second:evaluate'],
    }))
    store = StateStore(tmp_path)
    store.transition('first:evaluate', 'complete', attempt=1)
    monkeypatch.setattr(observer, 'query_compute_processes', lambda _: ())
    before = _snapshot(tmp_path)

    partial = observe(tmp_path, heartbeat_max_age=180)

    assert partial['health'] != 'complete'
    assert partial['completed_runs'] == 1
    assert partial['expected_runs'] == 2
    assert _snapshot(tmp_path) == before

    store.transition('second:evaluate', 'complete', attempt=1)
    complete = observe(tmp_path, heartbeat_max_age=180)
    assert complete['health'] == 'complete'


def test_subprocess_runner_refreshes_heartbeat_while_child_lives(
        tmp_path, monkeypatch):
    from tools.optimization import run_campaign

    records = []
    sleeps = []

    class Process:
        pid = 123

        def __init__(self):
            self.values = iter((None, None, 0))

        def poll(self):
            return next(self.values)

    monkeypatch.setattr(
        run_campaign, '_atomic_json',
        lambda path, value: records.append(dict(value)))
    monkeypatch.setattr(
        run_campaign.time, 'sleep', lambda seconds: sleeps.append(seconds))
    runner = run_campaign.SubprocessStageRunner(
        tmp_path, tmp_path / 'candidates.json', heartbeat_interval=7)

    returncode = runner.wait_with_heartbeat(
        Process(), tmp_path / 'heartbeat.json',
        {'stage_id': 'fixture:evaluate', 'attempt': 1})

    assert returncode == 0
    assert len(records) == 3
    assert sleeps == [7, 7]
    assert records[-1]['phase'] == 'exited'
