import pytest


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
            'boot_id': 'boot', 'timestamp': '2026-08-27T00:00:00+00:00',
            'device_index': 0, 'allowed_pids': [123],
        },
    )
    assert result['protocol'] == {
        'batch_size': 1,
        'warmup': 50,
        'iterations': 3,
        'timer': 'torch.cuda.Event',
        'synchronize': True,
        'scope': 'full_topdown_model',
    }
    assert set(result['modes']) == {'flip', 'no_flip'}
    assert result['gpu_lease']['stage_id'] == 'full-s-v1:latency'
