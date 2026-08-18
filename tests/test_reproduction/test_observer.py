import json
from pathlib import Path


def test_observer_cannot_change_campaign_state(tmp_path):
    from mambapose_repro.observe import observe
    from mambapose_repro.state import StateStore

    store = StateStore(tmp_path)
    store.transition('fixture', 'running', attempt=1)
    before = (tmp_path / 'state.json').read_bytes()
    status = observe(tmp_path, heartbeat_max_age=180)
    assert (tmp_path / 'state.json').read_bytes() == before
    assert (tmp_path / 'status.json').is_file()
    assert status['health'] in {'starting', 'failed'}


def test_fresh_live_heartbeat_is_running(tmp_path):
    from datetime import datetime, timezone
    import os

    from mambapose_repro.observe import observe
    from mambapose_repro.state import StateStore

    StateStore(tmp_path).transition('fixture', 'running', attempt=1)
    (tmp_path / 'heartbeat.json').write_text(json.dumps({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'run_id': 'fixture',
        'phase': 'train',
        'pid': os.getpid(),
        'log_bytes': 100,
    }))
    status = observe(tmp_path, heartbeat_max_age=180)
    assert status['health'] == 'running'
    assert status['process_alive'] is True


def test_observer_never_invokes_controller_process_operations():
    source = Path('mambapose_repro/observe.py').read_text()
    forbidden = ('Popen(', 'subprocess.run(', 'os.kill(', 'systemctl')
    assert all(token not in source for token in forbidden)


def test_observer_requires_every_manifest_run_before_complete(tmp_path):
    from mambapose_repro.observe import observe
    from mambapose_repro.state import StateStore

    store = StateStore(tmp_path)
    store.transition('first', 'complete', attempt=1)

    partial = observe(tmp_path, expected_run_ids=('first', 'second'))

    assert partial['health'] != 'complete'
    assert partial['completed_runs'] == 1
    assert partial['expected_runs'] == 2

    store.transition('second', 'complete', attempt=1)
    complete = observe(tmp_path, expected_run_ids=('first', 'second'))
    assert complete['health'] == 'complete'
