import pytest
from pathlib import Path
from datetime import datetime, timezone


class FakeTimer:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1

    def measure_ms(self, callable_):
        callable_()
        return next(self.samples)


def test_latency_summary_reports_median_and_interpolated_tails():
    from mambapose_opt.latency import LatencySummary

    summary = LatencySummary.from_samples_ms([1.0, 1.1, 1.2, 9.0])
    assert summary.median_ms == pytest.approx(1.15)
    assert summary.p90_ms == pytest.approx(6.66)
    assert summary.p95_ms == pytest.approx(7.83)
    assert summary.sample_count == 4


def test_measure_latency_uses_injected_timer_without_cuda():
    from mambapose_opt.latency import measure_latency

    calls = []
    timer = FakeTimer([1.0, 1.1, 1.2, 1.3])
    summary = measure_latency(
        lambda: calls.append('call'), warmup=3, repeats=4, timer=timer)

    assert len(calls) == 7
    assert timer.sync_count == 1
    assert summary.median_ms == pytest.approx(1.15)


@pytest.mark.parametrize('samples', [
    [1.0, -0.1],
    [1.0, float('nan')],
    [1.0, float('inf')],
])
def test_latency_rejects_negative_or_nonfinite_samples(samples):
    from mambapose_opt.latency import LatencyError, LatencySummary

    with pytest.raises(LatencyError, match='finite non-negative'):
        LatencySummary.from_samples_ms(samples)


def test_latency_protocol_requires_batch_one_both_flip_modes_and_lease():
    from mambapose_opt.latency import build_latency_result

    result = build_latency_result(
        flip=[1.0, 1.2, 1.1],
        no_flip=[0.8, 0.9, 1.0],
        warmup=50,
        repeats=3,
        gpu_lease={
            'stage_id': 'full-s-v1:latency', 'pid': 123,
            'boot_id': '11111111-1111-1111-1111-111111111111',
            'timestamp': '2026-08-27T00:00:00+00:00',
            'device_index': 0, 'allowed_pids': [123],
            'lease_id': '7' * 64,
        },
    )
    assert result['protocol'] == {
        'batch_size': 1,
        'warmup': 50,
        'iterations': 3,
        'timer': 'torch.cuda.Event',
        'synchronize': True,
        'scope': 'full_topdown_model',
        'lease_max_age_seconds': 300,
        'lease_max_future_skew_seconds': 30,
    }
    assert set(result['modes']) == {'flip', 'no_flip'}
    assert result['gpu_lease']['stage_id'] == 'full-s-v1:latency'


@pytest.mark.parametrize('change', [
    {'boot_id': 'not-a-uuid'},
    {'timestamp': '2026-08-27T00:00:00'},
    {'device_index': True},
    {'allowed_pids': [999]},
])
def test_gpu_lease_validation_rejects_malformed_provenance(change):
    from mambapose_opt.latency import LatencyError, validate_gpu_lease

    lease = {
        'stage_id': 'fixture:latency', 'pid': 123,
        'boot_id': '11111111-1111-1111-1111-111111111111',
        'timestamp': '2026-08-27T00:00:00+00:00',
        'device_index': 0, 'allowed_pids': [123],
        'lease_id': '7' * 64,
    }
    with pytest.raises(LatencyError):
        validate_gpu_lease({**lease, **change})


def test_latency_admission_rejects_unlocked_lease_before_model_or_cuda(
        tmp_path, monkeypatch):
    import hashlib
    import json
    import tools.optimization.measure_latency as tool
    from mambapose_opt.schema import CandidateSpec

    __import__('subprocess').run(['git', 'init', '-q'], cwd=tmp_path, check=True)

    (tmp_path / 'config.py').write_text('model = dict()')
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'checkpoint')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/inventory.json').write_text('{}')
    lock = tmp_path / 'gpu.lock'
    lock.write_text(json.dumps({
        'stage_id': 'fixture:latency', 'pid': 123,
        'boot_id': '11111111-1111-1111-1111-111111111111',
        'timestamp': '2026-08-27T00:00:00+00:00',
        'device_index': 0, 'allowed_pids': [123],
        'lease_id': '7' * 64,
    }))
    candidate = CandidateSpec.from_dict({
        'id': 'fixture', 'route': 'accuracy-first', 'kind': 'float',
        'config': 'config.py', 'checkpoint': 'model.pth',
        'checkpoint_sha256': hashlib.sha256(b'checkpoint').hexdigest(),
        'seed': 0, 'features': {},
    })
    monkeypatch.setattr(tool, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(tool, '_git_commit', lambda: 'd' * 40)
    monkeypatch.setattr(tool, '_canonical_gpu_lock', lambda: lock)
    monkeypatch.setenv('MAMBAPOSE_PHYSICAL_DEVICE_INDEX', '0')
    monkeypatch.setattr(
        tool.Config, 'fromfile', lambda *args: (_ for _ in ()).throw(
            AssertionError('model/config initialization happened too early')))

    with pytest.raises(ValueError, match='actively held'):
        tool.measure_candidate(candidate, warmup=50, repeats=200)


def test_active_gpu_lease_checks_boot_device_and_live_descendant(
        tmp_path, monkeypatch):
    import fcntl
    import json
    import os
    import tools.optimization.measure_latency as tool

    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    lock = tmp_path / 'gpu.lock'
    lease = {
        'stage_id': 'fixture:latency', 'pid': os.getpid(),
        'boot_id': boot_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'device_index': 2, 'allowed_pids': [os.getpid()],
        'lease_id': '7' * 64,
    }
    lock.write_text(json.dumps(lease))
    monkeypatch.setattr(tool, '_canonical_gpu_lock', lambda: lock)
    with lock.open('r+') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert tool._active_gpu_lease('fixture', 2) == lease
        owner.seek(0)
        owner.truncate()
        json.dump({**lease, 'boot_id': '11111111-1111-1111-1111-111111111111'}, owner)
        owner.flush()
        with pytest.raises(ValueError, match='different boot'):
            tool._active_gpu_lease('fixture', 2)


@pytest.mark.parametrize('timestamp', [
    '2000-01-01T00:00:00+00:00',
    '2999-01-01T00:00:00+00:00',
])
def test_active_gpu_lease_rejects_stale_or_future_timestamp(
        tmp_path, monkeypatch, timestamp):
    import fcntl
    import json
    import os
    import tools.optimization.measure_latency as tool

    lock = tmp_path / 'gpu.lock'
    lock.write_text(json.dumps({
        'stage_id': 'fixture:latency', 'pid': os.getpid(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'timestamp': timestamp, 'device_index': 2,
        'allowed_pids': [os.getpid()],
        'lease_id': '7' * 64,
    }))
    monkeypatch.setattr(tool, '_canonical_gpu_lock', lambda: lock)
    with lock.open('r+') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='timestamp|fresh'):
            tool._active_gpu_lease('fixture', 2)


def test_active_gpu_lease_rejects_exactly_301_seconds_without_heartbeat(
        tmp_path, monkeypatch):
    import fcntl
    import json
    import os
    import tools.optimization.measure_latency as tool

    start = datetime(2026, 8, 27, tzinfo=timezone.utc)
    lock = tmp_path / 'gpu.lock'
    lock.write_text(json.dumps({
        'stage_id': 'fixture:latency', 'pid': os.getpid(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'timestamp': start.isoformat(), 'device_index': 2,
        'allowed_pids': [os.getpid()],
        'lease_id': '7' * 64,
    }))
    monkeypatch.setattr(tool, '_canonical_gpu_lock', lambda: lock)
    with lock.open('r+') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='stale'):
            tool._active_gpu_lease(
                'fixture', 2,
                now=lambda: start + __import__('datetime').timedelta(
                    seconds=301))
