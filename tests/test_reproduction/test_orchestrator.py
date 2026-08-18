from pathlib import Path

import pytest


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self.outcomes)


def test_identical_transient_failure_exhausts_budget(tmp_path):
    from mambapose_repro.orchestrator import (
        AttemptOutcome, RetryExhausted, run_until_terminal)
    from mambapose_repro.state import StateStore

    runner = FakeRunner([
        AttemptOutcome(75, 'network-timeout'),
        AttemptOutcome(75, 'network-timeout'),
        AttemptOutcome(75, 'network-timeout'),
    ])
    with pytest.raises(RetryExhausted, match='network-timeout'):
        run_until_terminal(
            'fixture', runner, StateStore(tmp_path), max_attempts=3,
            delays=(0, 0, 0))
    assert runner.calls == 3
    state = StateStore(tmp_path).read()
    assert state['runs']['fixture']['attempt'] == 3
    assert state['runs']['fixture']['failure_fingerprint'] == 'network-timeout'


def test_permanent_failure_stops_without_retry(tmp_path):
    from mambapose_repro.orchestrator import (
        AttemptOutcome, PermanentFailure, run_until_terminal)
    from mambapose_repro.state import StateStore

    runner = FakeRunner([AttemptOutcome(78, 'invalid-config')])
    with pytest.raises(PermanentFailure, match='invalid-config'):
        run_until_terminal(
            'fixture', runner, StateStore(tmp_path), max_attempts=3,
            delays=(0, 0, 0))
    assert runner.calls == 1


def test_success_is_artifact_validated(tmp_path):
    from mambapose_repro.orchestrator import (
        AttemptOutcome, PermanentFailure, run_until_terminal)
    from mambapose_repro.state import StateStore

    runner = FakeRunner([AttemptOutcome(0, 'success', artifacts_valid=False)])
    with pytest.raises(PermanentFailure, match='artifact validation'):
        run_until_terminal(
            'fixture', runner, StateStore(tmp_path), max_attempts=1,
            delays=(0,))


def test_campaign_lock_rejects_concurrent_owner(tmp_path):
    from mambapose_repro.orchestrator import CampaignLock, ConcurrentCampaign

    with CampaignLock(tmp_path / 'campaign.lock'):
        with pytest.raises(ConcurrentCampaign):
            with CampaignLock(tmp_path / 'campaign.lock'):
                pass
