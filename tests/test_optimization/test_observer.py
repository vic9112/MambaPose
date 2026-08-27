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
