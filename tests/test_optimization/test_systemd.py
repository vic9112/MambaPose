from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SYSTEMD = ROOT / 'systemd'


def test_campaign_unit_has_durable_restart_and_group_shutdown():
    unit = (SYSTEMD / 'mambapose-optimization.service').read_text()

    assert 'Restart=on-failure' in unit
    assert 'RestartPreventExitStatus=78' in unit
    assert 'KillMode=control-group' in unit
    assert 'PYTHONNOUSERSITE=1' in unit
    assert 'CUDA_VISIBLE_DEVICES=0' in unit
    assert 'tools/optimization/run_campaign.py --run' in unit


def test_observer_timer_has_no_controller_authority():
    service = (
        SYSTEMD / 'mambapose-optimization-observer.service').read_text()
    timer = (SYSTEMD / 'mambapose-optimization-observer.timer').read_text()

    assert 'Type=oneshot' in service
    assert 'tools/optimization/observe.py' in service
    assert 'run_campaign.py' not in service
    assert 'OnUnitActiveSec=60s' in timer
    assert 'Persistent=true' in timer


def test_installer_fixture_verifies_units_without_installing(tmp_path):
    result = subprocess.run(
        ['bash', 'tools/optimization/install_user_service.sh',
         '--fixture-smoke'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'fixture systemd verification passed' in result.stdout
