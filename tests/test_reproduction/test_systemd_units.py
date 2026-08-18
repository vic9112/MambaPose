from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYSTEMD = ROOT / 'systemd'


def test_campaign_unit_has_bounded_restart_and_permanent_stop():
    unit = (SYSTEMD / 'mambapose-reproduction.service').read_text()
    assert 'Restart=on-failure' in unit
    assert 'RestartPreventExitStatus=78' in unit
    assert 'StartLimitIntervalSec=infinity' in unit
    assert 'StartLimitBurst=3' in unit
    assert 'KillMode=control-group' in unit
    assert 'PYTHONNOUSERSITE=1' in unit
    assert '/home/vicchen/workspace/MambaPose/.venv/bin/python' in unit
    assert 'tools/reproduction/run_campaign.py --run' in unit


def test_observer_is_oneshot_timer_and_has_no_controller_authority():
    service = (SYSTEMD / 'mambapose-observer.service').read_text()
    timer = (SYSTEMD / 'mambapose-observer.timer').read_text()
    assert 'Type=oneshot' in service
    assert 'tools/reproduction/observe.py' in service
    assert 'run_campaign.py' not in service
    assert 'OnUnitActiveSec=60s' in timer
    assert 'Persistent=true' in timer


def test_installer_checks_and_records_linger():
    source = (ROOT / 'tools/reproduction/install_user_service.sh').read_text()
    assert 'loginctl show-user' in source
    assert 'systemd-analyze --user verify' in source
    assert 'enable mambapose-reproduction.service' in source
    assert 'enable mambapose-observer.timer' in source
    assert 'durability_scope' in source
